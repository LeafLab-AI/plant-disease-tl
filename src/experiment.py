from __future__ import annotations
import json
import time
from pathlib import Path

import keras
import tensorflow as tf
import yaml

from src.datasets import DataManager
from src.models import ModelBuilder
from src.strategies import StrategyManager
from src.evaluate import ModelEvaluator

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SparseF1Score(keras.metrics.F1Score):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
        super().update_state(y_true_one_hot, y_pred, sample_weight)


class WarmupCallback(keras.callbacks.Callback):
    def __init__(self, warmup_epochs: int, base_lr: float):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr

    def on_epoch_begin(self, epoch, logs=None):
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            self.model.optimizer.learning_rate.assign(self.base_lr * (epoch + 1) / self.warmup_epochs)


class ExperimentRunner:
    def __init__(self, cfg: dict, model_name: str, strategy: str, data_manager: DataManager = None):
        self.cfg = cfg
        self.model_name = model_name
        self.strategy = strategy
        self.data_manager = data_manager or DataManager(cfg)
        self.run_dir = PROJECT_ROOT / "results" / f"{model_name}_{strategy}_s{cfg['seed']}"
        self.lr = float(cfg["training"]["lr"][strategy])

    def setup_environment(self) -> None:
        if self.cfg["training"]["mixed_precision"]:
            keras.mixed_precision.set_global_policy("mixed_float16")
        keras.utils.set_random_seed(self.cfg["seed"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.yaml").write_text(yaml.safe_dump(self.cfg, sort_keys=False))

    def _build_callbacks(self) -> list[keras.callbacks.Callback]:
        return [
            keras.callbacks.ModelCheckpoint(
                str(self.run_dir / "best.keras"), monitor="val_macro_f1", mode="max", save_best_only=True
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_macro_f1", mode="max", patience=6, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7, verbose=1
            ),
            keras.callbacks.CSVLogger(str(self.run_dir / "history.csv")),
            WarmupCallback(self.cfg["training"]["warmup_epochs"][self.strategy], self.lr),
        ]

    def run(self) -> tuple[Path, dict, keras.callbacks.History]:
        self.setup_environment()
        self.data_manager.prepare_data()

        train_ds = self.data_manager.get_dataset("train", training=True)
        val_ds = self.data_manager.get_dataset("val", training=False)
        test_ds = self.data_manager.get_dataset("test", training=False)
        weights = self.data_manager.get_class_weights() if self.cfg["training"]["class_weights"] else None

        builder = ModelBuilder(self.model_name, self.data_manager.num_classes, self.cfg["data"]["img_size"], self.cfg["training"]["dropout"])
        model, backbone = builder.build()
        
        StrategyManager.apply_strategy(backbone, self.strategy, self.cfg["training"]["unfreeze_fraction"])
        params = StrategyManager.summarize_trainable(model)

        model.compile(
            optimizer=keras.optimizers.AdamW(
                self.lr, weight_decay=self.cfg["training"]["weight_decay"], clipnorm=self.cfg["training"]["gradient_clip_norm"]
            ),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                SparseF1Score(average="macro", name="macro_f1")
            ],
        )

        t0 = time.time()
        history = model.fit(
            train_ds, validation_data=val_ds, epochs=self.cfg["training"]["epochs"][self.strategy],
            class_weight=weights, callbacks=self._build_callbacks(), verbose=2
        )
        train_sec = time.time() - t0

        best_model = keras.models.load_model(self.run_dir / "best.keras", compile=False)
        evaluator = ModelEvaluator(self.run_dir, self.data_manager.class_to_idx)
        eval_metrics = evaluator.evaluate(best_model, test_ds)

        metrics = {
            **eval_metrics, **params, "train_seconds": train_sec,
            "epochs_run": len(history.history["loss"]), "strategy": self.strategy, "model": self.model_name
        }
        (self.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        
        return self.run_dir, metrics, history