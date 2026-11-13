from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

import imagehash
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

# Popcount lookup for fast Hamming distances on packed perceptual hashes.
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        self.parent[self.find(i)] = self.find(j)


class DataManager:
    AUTOTUNE = tf.data.AUTOTUNE
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # 20 folds -> 14 train / 3 val / 3 test  (70 / 15 / 15)
    N_SPLITS = 20
    N_TRAIN_FOLDS = 14
    N_VAL_FOLDS = 3

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.raw_dir = self._resolve_path(cfg["data"]["raw_dir"])
        self.manifest_path = self._resolve_path(cfg["data"]["manifest"])
        self.assignments_path = self._resolve_path(cfg["data"]["assignments"])
        self.report_path = self._resolve_path(cfg["data"]["pipeline_report"])

        self.df: pd.DataFrame | None = None
        self.class_to_idx: Dict[str, int] = {}
        self.num_classes: int = 0

    # ------------------------------------------------------------------ paths
    def _resolve_path(self, p: str | Path) -> Path:
        path = Path(p)
        return path if path.is_absolute() else self.PROJECT_ROOT / path

    def _display_path(self, p: Path) -> str:
        try:
            return p.relative_to(self.PROJECT_ROOT).as_posix()
        except ValueError:
            return str(p)

    def _to_rel(self, p: str) -> str:
        """Convert a legacy absolute path (possibly from another machine) to a raw_dir-relative one."""
        p = str(p).replace("\\", "/")
        path = Path(p)
        try:
            return path.relative_to(self.raw_dir).as_posix()
        except ValueError:
            pass
        parts = path.parts
        name = self.raw_dir.name
        if name in parts:
            idx = len(parts) - 1 - parts[::-1].index(name)
            return Path(*parts[idx + 1:]).as_posix()
        return (Path(path.parent.name) / path.name).as_posix()  # label/filename fallback

    def _attach_paths(self, df: pd.DataFrame) -> pd.DataFrame:
        """Makes the 'path' column valid on *this* machine. Only 'rel_path' is persisted."""
        df = df.copy()
        if "rel_path" not in df.columns:
            df["rel_path"] = [self._to_rel(p) for p in df["path"]]
        df["path"] = [str(self.raw_dir / r) for r in df["rel_path"]]
        return df

    def _check_files_exist(self, df: pd.DataFrame) -> None:
        missing = [p for p in df["path"] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images listed in the splits are missing under {self.raw_dir}. "
                f"First missing: {missing[0]}. Check data.raw_dir in the config."
            )

    def prepare_data(self, force: bool = False) -> None:
        """Loads existing splits or builds them from scratch (idempotent)."""
        if self.df is not None and not force:
            return

        built = force or not (self.assignments_path.exists() and self.manifest_path.exists())
        if not built:
            self.df = self._attach_paths(pd.read_csv(self.assignments_path))
        else:
            self.df = self._build_manifest()
            self.df = self._deduplicate()
            self.df = self._stratified_group_split()

        # Idempotent, so split files cached before cleaning existed are upgraded on load.
        self.df, cleaning = self._clean(self.df)
        if built or cleaning["removed_total"] > 0:
            self.assignments_path.parent.mkdir(parents=True, exist_ok=True)
            self.df.drop(columns=["path"]).to_csv(self.assignments_path, index=False)
            self._update_report(cleaning)

        self._check_files_exist(self.df)
        classes = sorted(self.df["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.num_classes = len(self.class_to_idx)

    @staticmethod
    def _probe(img_path: Path, min_side: int, hash_size: int):
        """Returns a record dict, or 'small' / 'corrupt'."""
        try:
            with Image.open(img_path) as im:
                w, h = im.size
                if min(w, h) < min_side:
                    return "small"
                phash = str(imagehash.phash(im, hash_size=hash_size))
        except Exception:
            return "corrupt"
        return {"width": w, "height": h, "phash": phash}

    def _build_manifest(self) -> pd.DataFrame:
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"raw_dir does not exist: {self.raw_dir}")

        exts = {e.lower() for e in self.cfg["data"]["image_exts"]}
        min_side = self.cfg["data"]["min_side"]
        dedup_cfg = self.cfg["data"]["dedup"]
        workers = self.cfg["data"].get("num_workers", min(8, os.cpu_count() or 1))

        files = [p for p in sorted(self.raw_dir.rglob("*")) if p.suffix.lower() in exts]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda p: self._probe(p, min_side, dedup_cfg["hash_size"]), files))

        records, skipped_small, skipped_corrupt = [], 0, 0
        for p, r in zip(files, results):
            if r == "small":
                skipped_small += 1
            elif r == "corrupt":
                skipped_corrupt += 1
            else:
                records.append({
                    "rel_path": p.relative_to(self.raw_dir).as_posix(),
                    "label": p.parent.name, **r,
                })

        df = pd.DataFrame(records)
        if df.empty:
            raise RuntimeError(f"No images found. Skipped {skipped_small} small, {skipped_corrupt} corrupt.")

        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.manifest_path, index=False)

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps({
            "raw_dir": self._display_path(self.raw_dir),
            "num_images": int(len(df)),
            "num_classes": int(df["label"].nunique()),
            "class_counts": {k: int(v) for k, v in df["label"].value_counts().items()},
            "skipped_small": skipped_small,
            "skipped_corrupt": skipped_corrupt,
            "dedup": dedup_cfg,
            "manifest_sha256": hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
        }, indent=2))
        return self._attach_paths(df)

    def _deduplicate(self) -> pd.DataFrame:
        """Groups near-duplicate images (Hamming distance on pHash) so they can't leak across splits."""
        threshold = self.cfg["data"]["dedup"]["hamming_threshold"]
        codes = np.stack([np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in self.df["phash"]])

        n = len(codes)
        uf = _UnionFind(n)
        for i in range(n - 1):
            # only compare with later rows (j > i): halves the work; packed bytes shrink it 8x
            dists = _POPCOUNT[codes[i + 1:] ^ codes[i]].sum(axis=1, dtype=np.int32)
            for j in np.flatnonzero(dists <= threshold):
                uf.union(i, i + 1 + int(j))

        df = self.df.copy()
        df["group"] = [uf.find(i) for i in range(n)]
        return df

    def _clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Cleaning defined in notebooks/00_data_setup, applied identically for every model.

        * drop_conflicting_groups: a near-duplicate group whose images carry different labels
          contains at least one wrong label, so the whole group is removed.
        * one_per_group: keep only the first image (manifest order) of each near-duplicate
          group, so duplicates are not over-weighted.

        Runs *after* the group split: kept images keep their split, and since groups never
        cross splits no leakage is possible.
        """
        c = self.cfg["data"].get("clean", {})
        n_before = len(df)
        conflicting = duplicates = 0

        if c.get("drop_conflicting_groups", True):
            ok = df.groupby("group")["label"].transform("nunique") == 1
            conflicting = int((~ok).sum())
            df = df[ok]
        if c.get("one_per_group", True):
            n = len(df)
            df = df.drop_duplicates(subset="group", keep="first")
            duplicates = n - len(df)

        df = df.reset_index(drop=True)
        return df, {
            "removed_conflicting_label_images": conflicting,
            "removed_near_duplicates": duplicates,
            "removed_total": n_before - len(df),
            "images_after_cleaning": len(df),
        }

    def _update_report(self, cleaning: dict) -> None:
        report = json.loads(self.report_path.read_text()) if self.report_path.exists() else {}
        report["cleaning"] = {**self.cfg["data"].get("clean", {}), **cleaning}
        report["split_counts"] = {k: int(v) for k, v in self.df["split"].value_counts().items()}
        report["class_counts_after_cleaning"] = {
            k: int(v) for k, v in self.df["label"].value_counts().items()
        }
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(report, indent=2))

    def _stratified_group_split(self) -> pd.DataFrame:
        df = self.df.copy()
        seed = self.cfg["data"].get("split_seed", self.cfg["seed"])
        sgkf = StratifiedGroupKFold(n_splits=self.N_SPLITS, shuffle=True, random_state=seed)

        split = np.empty(len(df), dtype=object)
        for i, (_, test_idx) in enumerate(sgkf.split(df, y=df["label"], groups=df["group"])):
            if i < self.N_TRAIN_FOLDS:
                split[test_idx] = "train"
            elif i < self.N_TRAIN_FOLDS + self.N_VAL_FOLDS:
                split[test_idx] = "val"
            else:
                split[test_idx] = "test"
        df["split"] = split

        assert df["split"].notna().all(), "Images missing split assignment."
        return df

    def dataset_fingerprint(self) -> str:
        """SHA-256 of the split file: lets MLflow runs prove they used identical data."""
        self.prepare_data()
        return hashlib.sha256(self.assignments_path.read_bytes()).hexdigest()

    def get_class_weights(self) -> Dict[int, float]:
        self.prepare_data()
        counts = (
            self.df[self.df["split"] == "train"]["label"]
            .map(self.class_to_idx).value_counts().to_dict()
        )
        total = sum(counts.values())
        return {int(i): total / (self.num_classes * c) for i, c in counts.items()}

    def _build_augmentation_pipeline(self) -> keras.Sequential:
        a = self.cfg["augmentation"]
        seed = self.cfg["seed"]
        layers = []
        if a["flip"] in ("horizontal", "horizontal_and_vertical"):
            layers.append(keras.layers.RandomFlip(a["flip"], seed=seed))
        layers.append(keras.layers.RandomRotation(a["rotation"], fill_mode=a["fill_mode"], seed=seed))
        layers.append(keras.layers.RandomZoom(a["zoom"], fill_mode=a["fill_mode"], seed=seed))
        layers.append(keras.layers.RandomContrast(a["contrast"], value_range=(0, 255), seed=seed))
        return keras.Sequential(layers, name="augmentation_pipeline")

    def get_dataset(self, split: str, training: bool) -> tf.data.Dataset:
        self.prepare_data()

        sub = self.df[self.df["split"] == split]
        paths = sub["path"].astype(str).to_numpy()
        labels = sub["label"].map(self.class_to_idx).to_numpy(dtype=np.int32)
        size = self.cfg["data"]["img_size"]

        loader_cfg = self.cfg["loader"]
        num_calls = self.AUTOTUNE if loader_cfg["num_parallel_calls"] == "auto" else int(loader_cfg["num_parallel_calls"])
        prefetch_buf = self.AUTOTUNE if loader_cfg["prefetch"] == "auto" else int(loader_cfg["prefetch"])

        def _load(path, label):
            img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
            img = tf.image.resize(img, (size, size), method="bilinear")  # float32, 0-255
            img.set_shape((size, size, 3))
            return img, label

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            # Buffer = whole dataset: (path, label) pairs are tiny, and it gives a uniform shuffle.
            ds = ds.shuffle(max(loader_cfg["shuffle_buffer"], len(paths)),seed=self.cfg["seed"], reshuffle_each_iteration=True)

        ds = ds.map(_load, num_parallel_calls=num_calls)
        ds = ds.batch(loader_cfg["batch_size"], drop_remainder=False)

        if training:
            aug = self._build_augmentation_pipeline()
            # Augment per *batch* (vectorised); random layers still sample per image.
            # training=True is explicit so augmentation can never be silently skipped.
            ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=num_calls)

        return ds.prefetch(prefetch_buf)