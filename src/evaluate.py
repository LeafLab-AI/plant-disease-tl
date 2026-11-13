from __future__ import annotations

import json
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


class ModelEvaluator:
    def __init__(self, run_dir: Path, class_to_idx: dict[str, int]):
        self.run_dir = run_dir
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.n_classes = len(class_to_idx)
        self.labels = [self.idx_to_class[i] for i in range(self.n_classes)]

    def evaluate(self, model: keras.Model, test_ds: tf.data.Dataset) -> dict:
        y_true, y_prob = [], []
        for x, y in test_ds:
            y_prob.append(model.predict(x, verbose=0))
            y_true.append(y.numpy())
            
        y_true = np.concatenate(y_true)
        y_prob = np.concatenate(y_prob)
        y_pred = y_prob.argmax(axis=1)

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=self.labels, output_dict=True)
        auc_macro = float(roc_auc_score(
            keras.utils.to_categorical(y_true, self.n_classes), 
            y_prob, multi_class="ovr", average="macro"
        ))

        self._save_artifacts(cm, report, auc_macro)

        return {
            "macro_f1": report["macro avg"]["f1-score"],
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "accuracy": report["accuracy"],
            "auc_macro_ovr": auc_macro,
        }

    def _save_artifacts(self, cm: np.ndarray, report: dict, auc: float) -> None:
        (self.run_dir / "classification_report.json").write_text(json.dumps(report, indent=2))
        pd.DataFrame(cm, index=self.labels, columns=self.labels).to_csv(self.run_dir / "confusion_matrix.csv")

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