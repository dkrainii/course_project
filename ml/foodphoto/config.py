"""Shared settings of the dish-recognition experiment: paths, classes, image size, random seed.

The same code runs locally and in Google Colab. Data and checkpoints go to FOODPHOTO_DATA and
FOODPHOTO_RUNS when they are set (in Colab: local disk for images, Google Drive for runs), else
under ml/data/ and ml/models/ (not in git).
"""
from __future__ import annotations

import os
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FOODPHOTO_DATA", ML_ROOT / "data" / "foodphoto"))
RUNS_DIR = Path(os.environ.get("FOODPHOTO_RUNS", ML_ROOT / "models" / "foodphoto"))

FOOD101_DIR = DATA_DIR / "food-101"       # torchvision download: images/<class>/<id>.jpg, meta/
UA_DIR = DATA_DIR / "ua"                  # commons.py: ua/<dish>/<file>.jpg
UA_OWN_DIR = DATA_DIR / "ua_own"          # our own photos: ua_own/<dish>/*.jpg (test only)
RESIZED_DIR = DATA_DIR / "resized"        # prepare.py: every image, shorter side RESIZE_TO
INDEX = DATA_DIR / "index.csv"            # prepare.py: path, label, source, split

REPORT_DIR = ML_ROOT / "reports" / "foodphoto"
FIGURES_DIR = REPORT_DIR / "figures"
RESULTS_DIR = REPORT_DIR / "results"      # JSON / CSV with every number quoted in the report
SOURCES_CSV = REPORT_DIR / "ua_sources.csv"  # author and licence of every Commons photo

SEED = 42
RESIZE_TO = 256      # stored size (shorter side); training crops IMAGE_SIZE out of it
IMAGE_SIZE = 224
VAL_PER_FOOD101_CLASS = 50   # carved out of the official Food-101 train split
UA_VAL_SHARE = 0.15
UA_TEST_SHARE = 0.20


def ensure_dirs() -> None:
    for path in (DATA_DIR, RUNS_DIR, FIGURES_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
