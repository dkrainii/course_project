"""Step 2. Food-101 + Ukrainian dishes → one resized image set with a fixed train/val/test split.

* Food-101 (101 classes, 1000 photos each; Bossard et al., 2014) is downloaded with torchvision.
  Its official test split (250 per class) is the test set; VAL_PER_FOOD101_CLASS photos of each
  class are moved from the official train split to validation.
* Ukrainian dishes come from commons.py (ua/<dish>/). They are split by a hash of the file
  name: UA_TEST_SHARE test, UA_VAL_SHARE validation, the rest train.
* Our own photos (ua_own/<dish>/, optional) only go to a separate test set "own": photos taken
  the way a Kolos user would, never seen in training.

Every image is stored once with its shorter side RESIZE_TO (JPEG): decoding small files keeps
the GPU busy on Colab's two CPU cores. The split is deterministic (hash of the file id), so a
rerun on another machine gives the same sets.

    python -m ml.foodphoto.prepare [--no-download] [--workers N]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

from . import config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def unit_hash(key: str) -> float:
    """Deterministic number in [0, 1) for a file id."""
    digest = hashlib.md5(f"{config.SEED}:{key}".encode("utf-8")).hexdigest()
    return int(digest, 16) / 16 ** 32


def food101_rows() -> list[dict]:
    meta = config.FOOD101_DIR / "meta"
    rows = []
    for official in ("train", "test"):
        ids = (meta / f"{official}.txt").read_text(encoding="utf-8").split()
        by_class: dict[str, list[str]] = {}
        for image_id in ids:
            by_class.setdefault(image_id.split("/")[0], []).append(image_id)
        for label, class_ids in by_class.items():
            # Validation: the VAL_PER_FOOD101_CLASS train photos with the smallest hash.
            val = set(sorted(class_ids, key=unit_hash)[:config.VAL_PER_FOOD101_CLASS]) if official == "train" else set()
            for image_id in class_ids:
                split = "test" if official == "test" else ("val" if image_id in val else "train")
                rows.append({"source": "food101", "label": label, "split": split,
                             "original": str(config.FOOD101_DIR / "images" / f"{image_id}.jpg"),
                             "path": f"food101/{image_id}.jpg"})
    return rows


def folder_rows(root: Path, source: str, fixed_split: str | None = None) -> list[dict]:
    rows = []
    if not root.is_dir():
        return rows
    for dish_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for image in sorted(p for p in dish_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            if fixed_split:
                split = fixed_split
            else:
                h = unit_hash(f"{dish_dir.name}/{image.name}")
                split = "test" if h < config.UA_TEST_SHARE else (
                    "val" if h < config.UA_TEST_SHARE + config.UA_VAL_SHARE else "train")
            rows.append({"source": source, "label": dish_dir.name, "split": split, "original": str(image),
                         "path": f"{source}/{dish_dir.name}/{image.stem}.jpg"})
    return rows


def resize_one(job: tuple[str, str]) -> str | None:
    original, target = job
    target_path = config.RESIZED_DIR / target
    if target_path.exists():
        return None
    try:
        with Image.open(original) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            scale = config.RESIZE_TO / min(img.size)
            if scale < 1:
                img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.BICUBIC)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(target_path, "JPEG", quality=90)
    except OSError as exc:
        return f"{original}: {exc}"
    return None


def prepare(download: bool = True, workers: int | None = None) -> list[dict]:
    config.ensure_dirs()
    if download and not (config.FOOD101_DIR / "meta" / "train.txt").exists():
        from torchvision.datasets import Food101
        print("[prepare] downloading Food-101 (≈5 GB)…", flush=True)
        Food101(root=str(config.DATA_DIR), download=True)

    rows = food101_rows() if (config.FOOD101_DIR / "meta").is_dir() else []
    rows += folder_rows(config.UA_DIR, "ua")
    rows += folder_rows(config.UA_OWN_DIR, "own", fixed_split="test")
    if not rows:
        raise SystemExit(f"No images found under {config.DATA_DIR}")

    workers = workers or os.cpu_count() or 2
    print(f"[prepare] resizing {len(rows):,} images with {workers} workers…", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        errors = [e for e in pool.map(resize_one, [(r["original"], r["path"]) for r in rows], chunksize=64) if e]
    for error in errors:
        print("[prepare] skipped", error, flush=True)
    broken = {e.split(": ")[0] for e in errors}
    rows = [r for r in rows if r["original"] not in broken]

    with open(config.INDEX, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "label", "source", "split"])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in rows)

    counts: dict[tuple[str, str], int] = {}
    for r in rows:
        counts[(r["source"], r["split"])] = counts.get((r["source"], r["split"]), 0) + 1
    for (source, split), n in sorted(counts.items()):
        print(f"[prepare] {source:8} {split:5} {n:7,}", flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    prepare(download=not args.no_download, workers=args.workers)
