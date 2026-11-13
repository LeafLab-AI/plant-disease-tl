from __future__ import annotations

import json
import time
from pathlib import Path

import keras
import tensorflow as tf
import yaml

from src.datasets import DataManager
from src.evaluate import ModelEvaluator
from src.models import ModelBuilder
from src.strategies import StrategyManager
from src.tracking import MLflowTracker

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


class StepProgress(keras.callbacks.Callback):
    """One line every `every` steps: fit(verbose=2) is silent until an epoch ends, which
    makes a slow epoch indistinguishable from a hang. Shows elapsed time and throughput."""

    def __init__(self, every: int, batch_size: int):
        super().__init__()
        self.every = every
        self.batch_size = batch_size

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch, self.t0 = epoch + 1, time.time()

    def on_train_batch_end(self, batch, logs=None):
        step = batch + 1
        if step == 1 or step % self.every == 0:
            dt = time.time() - self.t0
            total = self.params.get("steps") or "?"
            print(f"  epoch {self.epoch} step {step}/{total}  {dt:6.0f}s  "
                  f"{step * self.batch_size / max(dt, 1e-9):6.1f} img/s", flush=True)

    def on_test_begin(self, logs=None):
        self.t_val = time.time()
        print(f"  epoch {self.epoch} validating...", flush=True)

    def on_test_end(self, logs=None):
        print(f"  epoch {self.epoch} validation took {time.time() - self.t_val:.0f}s", flush=True)


class ExperimentRunner:
    def __init__(self, cfg: dict, model_name: str, strategy: str, data_manager: DataManager | None = None):
        self.cfg = cfg
        self.model_name = model_name
        self.strategy = strategy
        self.data_manager = data_manager or DataManager(cfg)
        self.tracker = MLflowTracker(cfg)
        self.run_name = f"{model_name}_{strategy}_s{cfg['seed']}"
        self.run_dir = PROJECT_ROOT / "results" / self.run_name
        self.lr = float(cfg["training"]["lr"][strategy])

    def setup_environment(self) -> None:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
        if self.cfg["training"]["mixed_precision"]:
            keras.mixed_precision.set_global_policy("mixed_float16")
        keras.utils.set_random_seed(self.cfg["seed"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.yaml").write_text(yaml.safe_dump(self.cfg, sort_keys=False))

    def _build_callbacks(self) -> list[keras.callbacks.Callback]:
        callbacks: list[keras.callbacks.Callback] = [
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
            StepProgress(
                int(self.cfg["training"].get("log_every_steps", 100)),
                int(self.cfg["loader"]["batch_size"]),
            ),
        ]
        mlflow_cb = self.tracker.epoch_callback()
        if mlflow_cb is not None:
            callbacks.append(mlflow_cb)
        return callbacks

    def run(self) -> tuple[Path, dict, keras.callbacks.History]:
        self.setup_environment()
        self.data_manager.prepare_data()

        tags = {
            "model": self.model_name,
            "strategy": self.strategy,
            "seed": self.cfg["seed"],
            "dataset_sha256": self.data_manager.dataset_fingerprint()[:16],
        }

        with self.tracker.start_run(run_name=self.run_name, tags=tags):
            self.tracker.log_params({
                "model": self.model_name,
                "strategy": self.strategy,
                "lr": self.lr,
                "dataset_sha256": self.data_manager.dataset_fingerprint(),
                **self.cfg,
            })

            train_ds = self.data_manager.get_dataset("train", training=True)
            val_ds = self.data_manager.get_dataset("val", training=False)
            test_ds = self.data_manager.get_dataset("test", training=False)
            weights = self.data_manager.get_class_weights() if self.cfg["training"]["class_weights"] else None

            builder = ModelBuilder(
                self.model_name,
                self.data_manager.num_classes,
                self.cfg["data"]["img_size"],
                self.cfg["training"]["dropout"],
            )
            model, backbone = builder.build(strategy=self.strategy)

            StrategyManager.apply_strategy(
                backbone,
                self.strategy,
                self.cfg["training"]["unfreeze_fraction"],
                freeze_bn=self.cfg["training"].get("freeze_bn", True),
            )
            params = StrategyManager.summarize_trainable(model)
            self.tracker.log_params(params)

            model.compile(
                optimizer=keras.optimizers.AdamW(
                    self.lr,
                    weight_decay=self.cfg["training"]["weight_decay"],
                    clipnorm=self.cfg["training"]["gradient_clip_norm"],
                ),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=[
                    keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                    SparseF1Score(average="macro", name="macro_f1"),
                ],
            )

            t0 = time.time()
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.cfg["training"]["epochs"][self.strategy],
                class_weight=weights,
                callbacks=self._build_callbacks(),
                verbose=2,
            )
            train_sec = time.time() - t0

            best_model = keras.models.load_model(
                self.run_dir / "best.keras", 
                compile=False,
                custom_objects={"SparseF1Score": SparseF1Score}
            )
            evaluator = ModelEvaluator(self.run_dir, self.data_manager.class_to_idx)
            eval_metrics = evaluator.evaluate(best_model, test_ds)

            metrics = {
                **eval_metrics,
                **params,
                "train_seconds": train_sec,
                "epochs_run": len(history.history["loss"]),
                "strategy": self.strategy,
                "model": self.model_name,
            }
            (self.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

            self.tracker.log_metrics({
                **{f"test_{k}": v for k, v in eval_metrics.items()},
                "train_seconds": train_sec,
                "epochs_run": len(history.history["loss"]),
            })
            self.tracker.log_artifacts(self.run_dir)

        return self.run_dir, metrics, history