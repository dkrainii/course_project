"""Step 7. Compact models for the web app.

The tuned LightGBM models from step 4 are accurate but large (2000+ boosting rounds, 120–150 MB
as text). The web app needs one answer per product, so it gets a dedicated compact model per
feature set: the same tuned tree shape, a larger learning rate and 200 rounds (≈14 MB, ≈1.5 ms
per product). It is trained on the training part, checked on validation and reported on test,
like every other model in the experiment.

The macro-only model is a fallback for products without a full label. It is weak (≈62 %), so the
app shows its grade only when the model is confident; the threshold is chosen on validation.

Output (read by `score_model.py` in the web app):
    <out>/label.txt.gz, <out>/macros.txt.gz   LightGBM boosters (text format, gzip)
    <out>/meta.json                           feature order, thresholds, test metrics

    python -m ml.nutriscore.export [--out instance/models/nutriscore]
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config
from . import modeling as M

ROUNDS = 200
LEARNING_RATE = 0.2
# Fallback (macros only): smallest confidence whose validation accuracy reaches this level.
MACROS_TARGET_ACCURACY = 0.88
# Label model: below this confidence the app still shows the grade, but the threshold is
# recorded so the app could hide doubtful answers too (0 = always answer).
LABEL_MIN_CONFIDENCE = 0.0
DEFAULT_OUT = config.ML_ROOT.parent / "instance" / "models" / "nutriscore"
FORMAT_VERSION = 1


def pick_threshold(y: np.ndarray, proba: np.ndarray, target: float) -> float:
    confidence, correct = proba.max(axis=1), proba.argmax(axis=1) == y
    for level in np.round(np.arange(0.40, 0.96, 0.01), 2):
        keep = confidence >= level
        if keep.sum() >= 100 and correct[keep].mean() >= target:
            return float(level)
    return 0.95


def coverage(y: np.ndarray, proba: np.ndarray, level: float) -> dict:
    keep = proba.max(axis=1) >= level
    pred = proba.argmax(axis=1)
    return {
        "min_confidence": level,
        "coverage": round(float(keep.mean()), 4),
        "accuracy": round(float((pred[keep] == y[keep]).mean()), 4) if keep.any() else None,
        "within_one": round(float((np.abs(pred[keep] - y[keep]) <= 1).mean()), 4) if keep.any() else None,
    }


def export(out: Path = DEFAULT_OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    df = M.load_dataset()
    meta: dict = {
        "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "grades": list(config.GRADES),
        "training": {"rounds": ROUNDS, "learning_rate": LEARNING_RATE,
                     "rows": int((df["split"] == "train").sum())},
        "models": {},
    }
    for feature_set, features in config.FEATURE_SETS.items():
        tuned = M.read_json(f"metrics_{feature_set}.json")["models"]["LightGBM"]["best_params"]
        X_train, y_train = M.split_xy(df, "train", features)
        X_val, y_val = M.split_xy(df, "val", features)
        X_test, y_test = M.split_xy(df, "test", features)

        model = M.make_lgbm(n_estimators=ROUNDS, learning_rate=LEARNING_RATE, subsample_freq=1, **tuned)
        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = round(time.perf_counter() - started, 1)

        val_proba, test_proba = model.predict_proba(X_val), model.predict_proba(X_test)
        level = (pick_threshold(y_val, val_proba, MACROS_TARGET_ACCURACY) if feature_set == "macros"
                 else LABEL_MIN_CONFIDENCE)
        text = model.booster_.model_to_string()
        path = out / f"{feature_set}.txt.gz"
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as fh:
            fh.write(text)

        test = M.evaluate(y_test, test_proba)
        meta["models"][feature_set] = {
            "file": path.name,
            "features": features,
            "required": [f for f in features if f != "fiber"],  # fibre may be missing, as in training
            "params": {**tuned, "n_estimators": ROUNDS, "learning_rate": LEARNING_RATE},
            "fit_seconds": fit_seconds,
            "size_mb": {"text": round(len(text.encode("utf-8")) / 1e6, 2),
                        "gzip": round(path.stat().st_size / 1e6, 2)},
            "min_confidence": level,
            "test": {k: test[k] for k in ("accuracy", "macro_f1", "qwk", "mae_grades", "within_one")},
            "test_at_threshold": coverage(y_test, test_proba, level),
            "val_at_threshold": coverage(y_val, val_proba, level),
            "latency": M.latency(model, X_test),
        }
        print(f"[export] {feature_set}: test accuracy {test['accuracy']}, macro-F1 {test['macro_f1']}, "
              f"{meta['models'][feature_set]['size_mb']['gzip']} MB gzip, threshold {level}", flush=True)

    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    M.write_json("export.json", meta)
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    export(parser.parse_args().out)
