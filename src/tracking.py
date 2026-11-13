from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import keras

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def flatten(d: Mapping[str, Any], prefix: str = "", sep: str = ".") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(flatten(v, key, sep))
        else:
            out[key] = v
    return out


class MLflowEpochLogger(keras.callbacks.Callback):
    """Streams per-epoch Keras logs (and the current learning rate) to the active MLflow run."""

    def __init__(self, mlflow_module):
        super().__init__()
        self._mlflow = mlflow_module

    def on_epoch_end(self, epoch, logs=None):
        metrics = {k: float(v) for k, v in (logs or {}).items()}
        try:
            metrics["learning_rate"] = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        except Exception:
            pass
        self._mlflow.log_metrics(metrics, step=epoch)


class MLflowTracker:
    """Thin, failure-tolerant wrapper around MLflow.

    Config (all optional, under `mlflow:` in base.yaml):
        enabled         : true
        experiment_name : plant-disease-tl
        tracking_uri    : sqlite:///<repo>/mlflow.db   (env MLFLOW_TRACKING_URI wins)
        log_model       : false   (upload best.keras as an artifact)
        tags            : {}      (extra tags added to every run)

    If MLflow is missing or the server is unreachable, tracking disables itself with a
    warning instead of crashing a long training run.
    """

    def __init__(self, cfg: dict):
        m = cfg.get("mlflow") or {}
        self.enabled: bool = bool(m.get("enabled", True))
        self.experiment_name: str = m.get("experiment_name", "plant-disease-tl")
        self.tracking_uri: str = (
            os.environ.get("MLFLOW_TRACKING_URI")
            or m.get("tracking_uri")
            or f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
        )
        self.log_model: bool = bool(m.get("log_model", False))
        self.extra_tags: dict = dict(m.get("tags") or {})
        self.run_id: str | None = None
        self._mlflow = None

    def _setup(self):
        if self._mlflow is not None:
            return self._mlflow
        try:
            import mlflow

            mlflow.set_tracking_uri(self.tracking_uri)
            if self.tracking_uri.startswith(("sqlite:", "file:")):
                client = mlflow.MlflowClient()
                if client.get_experiment_by_name(self.experiment_name) is None:
                    client.create_experiment(
                        self.experiment_name,
                        artifact_location=(PROJECT_ROOT / "mlartifacts").as_uri(),
                    )
            mlflow.set_experiment(self.experiment_name)
            self._mlflow = mlflow
        except Exception as exc:  
            warnings.warn(f"MLflow disabled: {exc}")
            self.enabled = False
        return self._mlflow

    @contextmanager
    def start_run(self, run_name: str, tags: Mapping[str, Any] | None = None) -> Iterator[Any]:
        if not self.enabled or self._setup() is None:
            yield None
            return
        all_tags = {**self.extra_tags, **{k: str(v) for k, v in (tags or {}).items()}}
        with self._mlflow.start_run(run_name=run_name, tags=all_tags) as run:
            self.run_id = run.info.run_id
            yield run  # an exception here marks the run FAILED automatically

    def log_params(self, params: Mapping[str, Any]) -> None:
        if not self._active():
            return
        flat = {k: str(v)[:500] for k, v in flatten(params).items() if v is not None}
        self._mlflow.log_params(flat)

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if not self._active():
            return
        clean = {
            k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if clean:
            self._mlflow.log_metrics(clean, step=step)

    def log_artifacts(self, run_dir: Path) -> None:
        """Uploads every file in run_dir (checkpoint only if `log_model` is true)."""
        if not self._active():
            return
        run_dir = Path(run_dir)
        for f in sorted(run_dir.rglob("*")):
            if not f.is_file() or (f.suffix == ".keras" and not self.log_model):
                continue
            rel_parent = f.parent.relative_to(run_dir).as_posix()
            self._mlflow.log_artifact(str(f), artifact_path=None if rel_parent == "." else rel_parent)

    def epoch_callback(self) -> keras.callbacks.Callback | None:
        return MLflowEpochLogger(self._mlflow) if self._active() else None

    def _active(self) -> bool:
        return self.enabled and self._mlflow is not None and self._mlflow.active_run() is not None