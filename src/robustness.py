from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import f1_score

from src.datasets import DataManager
from src.evaluate import ModelEvaluator
from src.experiment import SparseF1Score
from src.tracking import MLflowTracker


class RobustnessEvaluator:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.cfg = yaml.safe_load((self.run_dir / "config.yaml").read_text())
        self.dm = DataManager(self.cfg)
        self.dm.prepare_data()

    @staticmethod
    def _apply_gaussian_noise(x: tf.Tensor, std: float = 20.0) -> tf.Tensor:
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=std, dtype=tf.float32)
        return tf.clip_by_value(x + noise, 0.0, 255.0)

    @staticmethod
    def _apply_contrast_shift(x: tf.Tensor, factor: float = 0.5) -> tf.Tensor:
        mean = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
        return tf.clip_by_value((x - mean) * factor + mean, 0.0, 255.0)

    @staticmethod
    def _apply_box_blur(x: tf.Tensor, ksize: int = 5) -> tf.Tensor:
        return tf.nn.avg_pool2d(x, ksize=ksize, strides=1, padding="SAME")

    def run(self) -> dict:
        model = keras.models.load_model(
    self.run_dir / "best.keras", 
    compile=False,
    custom_objects={"SparseF1Score": SparseF1Score}
        )
        test_ds = self.dm.get_dataset("test", training=False)

        perturbations = {
            "clean": lambda x: x,
            "gaussian_noise_std20": self._apply_gaussian_noise,
            "low_contrast_0.5x": self._apply_contrast_shift,
            "blur_k5": self._apply_box_blur,
        }

        results: dict[str, float] = {}
        for name, fn in perturbations.items():
            perturbed_ds = test_ds.map(lambda x, y: (fn(x), y), num_parallel_calls=tf.data.AUTOTUNE)
            y_true, y_prob = ModelEvaluator.predict(model, perturbed_ds)
            y_pred = y_prob.argmax(axis=1)
            results[f"rob_{name}_acc"] = float((y_true == y_pred).mean())
            results[f"rob_{name}_macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        (self.run_dir / "robustness.json").write_text(json.dumps(results, indent=2))

        tracker = MLflowTracker(self.cfg)
        with tracker.start_run(run_name=f"{self.run_dir.name}_robustness", tags={"stage": "robustness"}):
            tracker.log_metrics(results)
            tracker.log_artifacts(self.run_dir)

        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model robustness under corruptions.")
    parser.add_argument("--run", required=True, type=Path, help="Path to results/<run_name>")
    args = parser.parse_args()
    results = RobustnessEvaluator(args.run).run()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()