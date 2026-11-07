"""Download the Kaggle dataset and record its provenance (version, licence, checksum).

Usage:
    python scripts/download_data.py [--dataset owner/name] [--out data] [--force]

Needs Kaggle credentials: either the KAGGLE_API_TOKEN environment variable or
~/.kaggle/kaggle.json (never commit either; both are covered by .gitignore).

Result:
    data/raw/                 extracted images
    data/provenance.json      dataset ref, version, licence, archive SHA-256, file counts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATASET = "moazawadali/plant-leaf-diseases"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def kaggle(*args: str) -> None:
    """Run the Kaggle CLI from the current interpreter's environment."""
    cmd = [sys.executable, "-c", "from kaggle.cli import main; main()", *args]
    subprocess.run(cmd, check=True)  # noqa: S603 - fixed argv, no shell


def check_credentials() -> None:
    import os

    home = Path.home() / ".kaggle"
    if os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        return
    if (home / "kaggle.json").exists() or (home / "access_token").exists():
        return
    sys.exit(
        "No Kaggle credentials found.\n"
        "  Create a token at https://www.kaggle.com/settings (API section), then either\n"
        "  set KAGGLE_API_TOKEN, or save kaggle.json to " + str(home / "kaggle.json")
    )


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--force", action="store_true", help="re-download even if data/raw exists")
    args = p.parse_args()

    out: Path = args.out
    raw = out / "raw"
    if raw.exists() and any(raw.iterdir()) and not args.force:
        sys.exit(f"{raw} already populated; use --force to re-download.")
    check_credentials()
    out.mkdir(parents=True, exist_ok=True)

    # 1. Metadata (licence, title, ...) and the archive itself, kept zipped so we can hash it.
    kaggle("datasets", "metadata", args.dataset, "-p", str(out))
    kaggle("datasets", "download", args.dataset, "-p", str(out), "--force")

    archive = out / f"{args.dataset.split('/')[1]}.zip"
    if not archive.exists():
        zips = sorted(out.glob("*.zip"), key=lambda z: z.stat().st_mtime)
        if not zips:
            sys.exit("Download finished but no .zip found in output folder.")
        archive = zips[-1]

    digest = sha256(archive)
    print(f"SHA-256 {archive.name}: {digest}")

    # 2. Extract (zipfile strips absolute paths and '..' components).
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw)

    # 3. Provenance manifest.
    meta_file = out / "dataset-metadata.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    meta = meta.get("info", meta)  # kaggle>=2 nests fields under "info"
    images = [f for f in raw.rglob("*") if f.suffix.lower() in IMAGE_EXTS]
    per_folder = Counter(f.parent.relative_to(raw).as_posix() for f in images)

    provenance = {
        "dataset": args.dataset,
        "url": f"https://www.kaggle.com/datasets/{args.dataset}",
        "title": meta.get("title"),
        "licenses": meta.get("licenses"),
        "version_note": "Kaggle CLI downloads the latest version; this checksum pins what was used.",
        "archive": archive.name,
        "archive_sha256": digest,
        "archive_bytes": archive.stat().st_size,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_images": len(images),
        "images_per_folder": dict(sorted(per_folder.items())),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    archive.unlink()
    print(f"{len(images)} images in {len(per_folder)} folders -> {raw}")
    print(f"Provenance written to {out / 'provenance.json'}")


if __name__ == "__main__":
    main()
