from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import imagehash
from sklearn.model_selection import StratifiedGroupKFold


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

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.raw_dir = self._resolve_path(cfg["data"]["raw_dir"])
        self.manifest_path = self._resolve_path(cfg["data"]["manifest"])
        self.assignments_path = self._resolve_path(cfg["data"]["assignments"])
        self.report_path = self._resolve_path(cfg["data"]["pipeline_report"])
        
        self.df: pd.DataFrame | None = None
        self.class_to_idx: Dict[str, int] = {}
        self.num_classes: int = 0

    def _resolve_path(self, p: str | Path) -> Path:
        path = Path(p)
        return path if path.is_absolute() else self.PROJECT_ROOT / path

    def prepare_data(self) -> None:
        """Loads existing splits or builds them from scratch."""
        if self.assignments_path.exists() and self.manifest_path.exists():
            self.df = pd.read_csv(self.assignments_path)
        else:
            self.df = self._build_manifest()
            self.df = self._deduplicate()
            self.df = self._stratified_group_split()
            self.assignments_path.parent.mkdir(parents=True, exist_ok=True)
            self.df.to_csv(self.assignments_path, index=False)
            
        classes = sorted(self.df["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.num_classes = len(self.class_to_idx)

    def _build_manifest(self) -> pd.DataFrame:
        exts = {e.lower() for e in self.cfg["data"]["image_exts"]}
        min_side = self.cfg["data"]["min_side"]
        dedup_cfg = self.cfg["data"]["dedup"]

        records = []
        skipped_small, skipped_corrupt = 0, 0

        for img_path in sorted(self.raw_dir.rglob("*")):
            if img_path.suffix.lower() not in exts:
                continue
            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    if min(w, h) < min_side:
                        skipped_small += 1
                        continue
                    phash = str(imagehash.phash(im, hash_size=dedup_cfg["hash_size"]))
            except Exception:
                skipped_corrupt += 1
                continue

            records.append({
                "path": str(img_path), "label": img_path.parent.name,
                "width": w, "height": h, "phash": phash,
            })

        df = pd.DataFrame(records)
        if df.empty:
            raise RuntimeError(f"No images found. Skipped {skipped_small} small, {skipped_corrupt} corrupt.")

        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.manifest_path, index=False)

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps({
            "raw_dir": str(self.raw_dir),
            "num_images": int(len(df)),
            "num_classes": int(df["label"].nunique()),
            "class_counts": df["label"].value_counts().to_dict(),
            "skipped_small": skipped_small,
            "skipped_corrupt": skipped_corrupt,
            "dedup": dedup_cfg,
            "manifest_sha256": hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
        }, indent=2))
        return df

    def _deduplicate(self) -> pd.DataFrame:
        threshold = self.cfg["data"]["dedup"]["hamming_threshold"]
        hashes = np.array([
            np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8)) 
            for h in self.df["phash"]
        ], dtype=np.uint8)
        
        n = len(hashes)
        uf = _UnionFind(n)
        
        for i in range(n):
            dists = np.sum(hashes ^ hashes[i], axis=1)
            for j in np.where(dists <= threshold)[0]:
                if j > i:
                    uf.union(i, j)
                    
        df = self.df.copy()
        df["group"] = [uf.find(i) for i in range(n)]
        return df

    def _stratified_group_split(self) -> pd.DataFrame:
        df = self.df.copy()
        sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=self.cfg["seed"])
        df["split"] = ""
        
        for i, (_, test_idx) in enumerate(sgkf.split(df, y=df["label"], groups=df["group"])):
            if i < 14: split_name = "train"
            elif i < 17: split_name = "val"
            else: split_name = "test"
            df.iloc[test_idx, df.columns.get_loc("split")] = split_name
            
        assert df["split"].ne("").all(), "Images missing split assignment."
        return df

    def get_class_weights(self) -> Dict[int, float]:
        if self.df is None:
            self.prepare_data()
        counts = self.df[self.df["split"] == "train"]["label"].map(self.class_to_idx).value_counts().to_dict()
        total = sum(counts.values())
        return {i: total / (len(counts) * c) for i, c in counts.items()}

    def _build_augmentation_pipeline(self) -> tf.keras.Sequential:
        a = self.cfg["augmentation"]
        layers = []
        if a["flip"] == "horizontal":
            layers.append(tf.keras.layers.RandomFlip("horizontal"))
        elif a["flip"] == "horizontal_and_vertical":
            layers.append(tf.keras.layers.RandomFlip("horizontal_and_vertical"))
            
        layers.append(tf.keras.layers.RandomRotation(a["rotation"], fill_mode=a["fill_mode"]))
        layers.append(tf.keras.layers.RandomZoom(a["zoom"], fill_mode=a["fill_mode"]))
        return tf.keras.Sequential(layers, name="augmentation_pipeline")

    def get_dataset(self, split: str, training: bool) -> tf.data.Dataset:
        if self.df is None:
            self.prepare_data()

        sub = self.df[self.df["split"] == split]
        paths = sub["path"].values
        labels = np.array([self.class_to_idx[l] for l in sub["label"]], dtype=np.int32)
        size = self.cfg["data"]["img_size"]
        contrast_val = self.cfg["augmentation"]["contrast"]
        aug_pipeline = self._build_augmentation_pipeline() if training else None

        loader_cfg = self.cfg["loader"]
        num_calls = self.AUTOTUNE if loader_cfg["num_parallel_calls"] == "auto" else int(loader_cfg["num_parallel_calls"])
        prefetch_buf = self.AUTOTUNE if loader_cfg["prefetch"] == "auto" else int(loader_cfg["prefetch"])

        def _decode_resize(path: tf.Tensor) -> tf.Tensor:
            img = tf.io.read_file(path)
            img = tf.io.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, (size, size), method="bilinear")
            return tf.cast(img, tf.float32)

        def _map(path, label):
            img = _decode_resize(path)
            if training:
                img = aug_pipeline(img[None, ...])[0]
                img = tf.image.random_contrast(img / 255.0, 1 - contrast_val, 1 + contrast_val) * 255.0
            return img, label

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(loader_cfg["shuffle_buffer"], seed=self.cfg["seed"], reshuffle_each_iteration=True)

        ds = ds.map(_map, num_parallel_calls=num_calls)
        ds = ds.batch(loader_cfg["batch_size"], drop_remainder=False)
        return ds.prefetch(prefetch_buf)