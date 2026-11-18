"""Pack the reviewed Ukrainian photos (ua/ and ua_own/) into ua_photos.zip for Google Colab.

    python -m ml.foodphoto.pack   →  ml/data/foodphoto/ua_photos.zip (upload to My Drive/Kolos/foodphoto/)
"""
from __future__ import annotations

import zipfile

from . import config
from .prepare import IMAGE_SUFFIXES

OUT = config.DATA_DIR / "ua_photos.zip"


def pack() -> None:
    count = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as archive:  # JPEGs do not compress further
        for folder in (config.UA_DIR, config.UA_OWN_DIR):
            if not folder.is_dir():
                continue
            for path in sorted(folder.rglob("*")):
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    archive.write(path, path.relative_to(config.DATA_DIR).as_posix())
                    count += 1
    print(f"Wrote {count} photos to {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    pack()
