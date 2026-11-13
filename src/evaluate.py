from __future__ import annotations

import json
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


class ModelEvaluator:
    def __init__(self, run_dir: Path, class_to_idx: dict[str, int], ece_bins: int = 15):
        self.run_dir = Path(run_dir)
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.n_classes = len(class_to_idx)
        self.labels = [self.idx_to_class[i] for i in range(self.n_classes)]
        self.ece_bins = ece_bins

    @staticmethod
    def predict(model: keras.Model, ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
        """Single pass over the dataset. Returns (y_true, y_prob)."""
        infer = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
        y_true, y_prob = [], []
        for x, y in ds:
            y_prob.append(np.asarray(infer(x), dtype=np.float32))
            y_true.append(y.numpy())
        return np.concatenate(y_true), np.concatenate(y_prob)

    @staticmethod
    def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
        conf = y_prob.max(axis=1)
        correct = (y_prob.argmax(axis=1) == y_true).astype(np.float64)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
        return float(ece)

    def evaluate(self, model: keras.Model, test_ds: tf.data.Dataset) -> dict:
        y_true, y_prob = self.predict(model, test_ds)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # guard against fp rounding
        y_pred = y_prob.argmax(axis=1)
        all_labels = list(range(self.n_classes))

        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        report = classification_report(
            y_true, y_pred, labels=all_labels, target_names=self.labels,
            output_dict=True, zero_division=0,
        )

        try:
            auc_macro = float(roc_auc_score(
                np.eye(self.n_classes)[y_true], y_prob, multi_class="ovr", average="macro"
            ))
        except ValueError:  # a class is absent from the test split
            auc_macro = float("nan")

        metrics = {
            "macro_f1": report["macro avg"]["f1-score"],
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "accuracy": float((y_true == y_pred).mean()),
            "auc_macro_ovr": auc_macro,
            "ece": self.expected_calibration_error(y_true, y_prob, self.ece_bins),
            "n_test": int(len(y_true)),
        }
        if self.n_classes > 3:
            top3 = np.argpartition(-y_prob, 2, axis=1)[:, :3]
            metrics["top3_accuracy"] = float((top3 == y_true[:, None]).any(axis=1).mean())

        self._save_artifacts(cm, report, y_true, y_prob)
        return metrics

    def _save_artifacts(self, cm: np.ndarray, report: dict, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "classification_report.json").write_text(json.dumps(report, indent=2))
        pd.DataFrame(cm, index=self.labels, columns=self.labels).to_csv(self.run_dir / "confusion_matrix.csv")
        # Saved so robustness tests / significance tests / calibration plots never need to re-run inference.
        np.savez_compressed(self.run_dir / "predictions.npz", y_true=y_true, y_prob=y_prob.astype(np.float16))

        fig, ax = plt.subplots(figsize=(max(6, self.n_classes * 0.6), max(5, self.n_classes * 0.5)))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(self.n_classes))
        ax.set_xticklabels(self.labels, rotation=45, ha="right")
        ax.set_yticks(range(self.n_classes))
        ax.set_yticklabels(self.labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (test)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(self.run_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)