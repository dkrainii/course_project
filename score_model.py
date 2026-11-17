"""The trained Nutri-Score model inside the web app.

`ml/nutriscore/export.py` writes two compact LightGBM boosters and a meta.json (feature order,
confidence thresholds, test metrics) to `instance/models/nutriscore/`. This module loads them
and registers one callable with `ml_inference`, so the rest of the app only sees
"nutrition per 100 g → grade or None".

Which booster answers depends on what the product's label gives:

* sugars, saturated fat and salt known (fibre may be missing, as in training) → the label
  model, ≈88 % of grades right on the test set;
* only the four macronutrients → the macro model, and only when it is confident (threshold
  chosen on validation: ≈20 % of such products get a grade, ≈88 % of those grades right).
  Otherwise there is no grade: a wrong one would be worse than none.

LightGBM is imported lazily: without it, or without the exported files, the app runs exactly
as before (grades from Open Food Facts only).
"""
from __future__ import annotations

import gzip
import json
import logging
import math
from functools import lru_cache
from pathlib import Path

import ml_inference

logger = logging.getLogger(__name__)

LABEL, MACROS = "label", "macros"
CACHE_SIZE = 8192


class NutriScoreModel:
    def __init__(self, directory: Path):
        import lightgbm as lgb  # optional dependency, see requirements.txt
        import numpy as np

        self._np = np
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        self.grades: tuple[str, ...] = tuple(meta["grades"])
        self.specs: dict[str, dict] = meta["models"]
        self.boosters = {}
        for name in (LABEL, MACROS):
            with gzip.open(directory / self.specs[name]["file"], "rt", encoding="utf-8") as fh:
                self.boosters[name] = lgb.Booster(model_str=fh.read())
        # Lists of products repeat the same foods: cache by the exact feature values.
        self._predict = lru_cache(maxsize=CACHE_SIZE)(self._predict_uncached)

    def choose(self, features: dict[str, float | None]) -> str:
        required = self.specs[LABEL]["required"]
        return LABEL if all(features.get(f) is not None for f in required) else MACROS

    def probabilities(self, name: str, values: tuple[float, ...]):
        row = self._np.array([[math.nan if v is None else v for v in values]], dtype=float)
        return self.boosters[name].predict(row)[0]

    def _predict_uncached(self, name: str, values: tuple[float | None, ...]) -> str | None:
        proba = self.probabilities(name, values)
        best = int(proba.argmax())
        if float(proba[best]) < self.specs[name]["min_confidence"]:
            return None
        return self.grades[best]

    def __call__(self, features: dict[str, float | None]) -> str | None:
        name = self.choose(features)
        values = tuple(features.get(f) for f in self.specs[name]["features"])
        return self._predict(name, values)


def default_directory(app) -> Path:
    return Path(app.instance_path) / "models" / "nutriscore"


def load_score_model(app) -> bool:
    """Register the exported model if it is there. Config SCORE_MODEL_DIR: a path, or None to skip."""
    directory = app.config.get("SCORE_MODEL_DIR", default_directory(app))
    if not directory:
        return False
    directory = Path(directory)
    if not (directory / "meta.json").is_file():
        logger.info("Nutri-Score model not found in %s; only Open Food Facts grades are shown.", directory)
        return False
    try:
        model = NutriScoreModel(directory)
    except ImportError:
        logger.warning("LightGBM is not installed; the Nutri-Score model in %s is not used.", directory)
        return False
    except (OSError, KeyError, ValueError) as exc:
        logger.warning("Could not load the Nutri-Score model from %s: %s", directory, exc)
        return False
    ml_inference.register_score_model(model)
    logger.info("Nutri-Score model loaded from %s.", directory)
    return True
