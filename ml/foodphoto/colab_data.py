"""Colab, step 4 of the notebook: get the prepared images onto the local disk.

The first session unpacks ua_photos.zip, runs prepare.py (Food-101 download and resize) and
caches the result on Google Drive as prepared.tar; later sessions only unpack the cache. A cache
without Ukrainian photos (the first Colab attempt) is rebuilt.

    python -m ml.foodphoto.colab_data --drive /content/drive/MyDrive/Kolos/foodphoto
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

from . import config
from .prepare import prepare


def sources() -> Counter:
    if not config.INDEX.exists():
        return Counter()
    with open(config.INDEX, encoding="utf-8") as fh:
        return Counter(row["source"] for row in csv.DictReader(fh))


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def unpack_photos(photos: Path) -> None:
    """Copy the zip to the local disk first: reading a big file straight from Drive can fail."""
    local = config.DATA_DIR / "ua_photos.zip"
    print(f"[colab] copying {photos.name} ({photos.stat().st_size / 1e6:.0f} MB) to the local disk", flush=True)
    shutil.copyfile(photos, local)
    try:
        with zipfile.ZipFile(local) as archive:
            broken = archive.testzip()
            if broken:
                raise SystemExit(f"ua_photos.zip is damaged ({broken}): upload it to Drive again")
            for name in ("ua", "ua_own"):
                shutil.rmtree(config.DATA_DIR / name, ignore_errors=True)
            archive.extractall(config.DATA_DIR)
            print(f"[colab] unpacked {len(archive.namelist())} photos", flush=True)
    except zipfile.BadZipFile as exc:
        raise SystemExit(f"ua_photos.zip is not a complete zip ({exc}): wait for the upload to finish "
                         "or upload it to Drive again") from exc
    finally:
        local.unlink(missing_ok=True)


def main(drive: Path, workers: int, download: bool = True) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache, photos = drive / "prepared.tar", drive / "ua_photos.zip"
    print(f"[colab] free space: local disk {free_gb(config.DATA_DIR):.0f} GB, Drive {free_gb(drive):.1f} GB",
          flush=True)
    if not photos.exists():
        raise SystemExit(f"Upload ua_photos.zip to {drive} first")

    if cache.exists() and not sources():
        subprocess.run(["tar", "-xf", str(cache), "-C", str(config.DATA_DIR)], check=True)
    if not sources()["ua"]:
        if cache.exists():
            print("[colab] the cached data has no Ukrainian photos: rebuilding it", flush=True)
            cache.unlink()
        shutil.rmtree(config.RESIZED_DIR, ignore_errors=True)
        config.INDEX.unlink(missing_ok=True)
        unpack_photos(photos)
        prepare(download=download, workers=workers)
        if free_gb(drive) < 3:
            print("[colab] less than 3 GB free on Drive: the cache is not saved, the next session "
                  "prepares the data again", flush=True)
        else:
            subprocess.run(["tar", "-cf", str(cache), "-C", str(config.DATA_DIR), "resized", "index.csv"],
                           check=True)
    counts = sources()
    print(f"[colab] images: {dict(counts)}", flush=True)
    if not counts["ua"]:
        raise SystemExit("No Ukrainian photos in the prepared data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--drive", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    main(args.drive, args.workers)
