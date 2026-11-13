"""Colab runtime setup, called by the bootstrap cell at the top of every notebook.

When a notebook open in VS Code runs on a Colab kernel, the code executes on a fresh
Google VM that only sees what the bootstrap cell cloned from GitHub. This module does the
rest of the per-VM setup:

* installs the few packages Colab does not ship (requirements-colab.txt), once per VM;
* downloads the Kaggle dataset into data/ if it is not there yet;
* forwards optional MLflow credentials so runs survive the VM being recycled.

Secrets are read from, in order: environment variables, Colab Secrets
(``google.colab.userdata``), then an interactive prompt for required ones. Nothing is
ever written to the repository.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_INSTALLED_MARKER = Path("/tmp/.plant_disease_tl_deps")
_MLFLOW_VARS = ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")


def get_secret(name: str, required: bool = True) -> str | None:
    """Environment variable -> Colab Secret -> getpass prompt (only if required)."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        from google.colab import userdata

        value = userdata.get(name)
        if value:
            os.environ[name] = value
            return value
    except Exception:  # noqa: BLE001 - secret missing, access not granted, or not on Colab
        pass
    if not required:
        return None
    from getpass import getpass

    value = getpass(f"{name}: ").strip()
    os.environ[name] = value
    return value


def _install_dependencies() -> None:
    if _INSTALLED_MARKER.exists():
        return
    req = PROJECT_ROOT / "requirements-colab.txt"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=True)
    _INSTALLED_MARKER.touch()


def _ensure_dataset() -> None:
    data = PROJECT_ROOT / "data"
    raw = data / "raw"
    if raw.exists() and any(raw.iterdir()):
        return
    get_secret("KAGGLE_API_TOKEN")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "download_data.py"), "--out", str(data)],
        check=True,
        cwd=PROJECT_ROOT,
    )


def _configure_tracking() -> None:
    for var in _MLFLOW_VARS:
        get_secret(var, required=False)
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        print(
            "WARNING: no MLFLOW_TRACKING_URI secret: runs are logged to "
            f"{PROJECT_ROOT}/mlflow.db and results/ on the Colab VM, which are deleted when "
            "the runtime is recycled. Copy results/ elsewhere before disconnecting."
        )


def _git_commit() -> str:
    out = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "log", "-1", "--format=%h %s"],
        capture_output=True, text=True, check=False,
    )
    return out.stdout.strip() or "unknown"


def setup_runtime(download_data: bool = True) -> None:
    """Idempotent: safe to re-run after a kernel restart on the same VM."""
    _install_dependencies()
    if download_data:
        _ensure_dataset()
    _configure_tracking()

    try:
        import tensorflow as tf

        gpus = [g.name for g in tf.config.list_physical_devices("GPU")]
    except Exception:  # noqa: BLE001
        gpus = []
    print(f"Running commit : {_git_commit()}")
    print(f"GPU            : {gpus or 'none (Runtime > Change runtime type)'}")
