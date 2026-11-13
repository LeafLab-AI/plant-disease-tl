from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
import yaml

from src.experiment import SparseF1Score
from src.tracking import MLflowTracker


class ModelBenchmarker:
    def __init__(self, run_dir: Path, warmup_runs: int = 10, timed_runs: int = 100):
        self.run_dir = Path(run_dir)
        self.cfg = yaml.safe_load((self.run_dir / "config.yaml").read_text())
        self.img_size = int(self.cfg["data"]["img_size"])
        self.warmup_runs = warmup_runs
        self.timed_runs = timed_runs

    def _measure_tflite_size_mb(self, model: keras.Model) -> float:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_bytes = converter.convert()
        tflite_path = self.run_dir / "model_quant.tflite"
        tflite_path.write_bytes(tflite_bytes)
        return len(tflite_bytes) / (1024.0 * 1024.0)

    def run(self) -> dict:
        model = keras.models.load_model(
    self.run_dir / "best.keras", 
    compile=False,
    custom_objects={"SparseF1Score": SparseF1Score}
        )
        dummy = np.random.uniform(0, 255, size=(1, self.img_size, self.img_size, 3)).astype(np.float32)

        with tf.device("/CPU:0"):
            infer = tf.function(lambda x: model(x, training=False))
            for _ in range(self.warmup_runs):
                infer(dummy)

            latencies_ms = []
            for _ in range(self.timed_runs):
                t0 = time.perf_counter()
                infer(dummy)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        bench_metrics = {
            "cpu_latency_mean_ms": float(np.mean(latencies_ms)),
            "cpu_latency_p95_ms": float(np.percentile(latencies_ms, 95)),
            "keras_size_mb": (self.run_dir / "best.keras").stat().st_size / (1024.0 * 1024.0),
            "tflite_int8_size_mb": self._measure_tflite_size_mb(model),
        }

        out_path = self.run_dir / "benchmark.json"
        out_path.write_text(json.dumps(bench_metrics, indent=2))

        tracker = MLflowTracker(self.cfg)
        with tracker.start_run(run_name=f"{self.run_dir.name}_benchmark", tags={"stage": "benchmark"}):
            tracker.log_metrics(bench_metrics)
            tracker.log_artifacts(self.run_dir)

        return bench_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark trained model latency & footprint.")
    parser.add_argument("--run", required=True, type=Path, help="Path to results/<run_name>")
    args = parser.parse_args()
    results = ModelBenchmarker(args.run).run()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()