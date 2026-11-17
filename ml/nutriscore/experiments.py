"""Step 5. Side experiments with the best model family (LightGBM):

* ablation — which inputs matter: add one label field to the four macros, drop one field from
  the full label, add what a label does not have (food group, fruit/vegetable share);
* learning curves — how much data the models need;
* compact models — what the trained LightGBM loses when only its first k boosting rounds are
  kept (a smaller file and a faster answer for the web app).

The test part is used for the ablation and compact-model tables; learning curves are read on the
validation part.

    python -m ml.nutriscore.experiments [ablation] [learning] [compact]
"""
from __future__ import annotations

import sys
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config
from . import modeling as M

LGBM_FAST = {"n_estimators": 3000, "learning_rate": 0.1, "subsample_freq": 1}
PATIENCE = 50
LEARNING_SIZES = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000]
EXTRA = [config.GROUP_COLUMN, "fvl"]
COMPACT_ROUNDS = [25, 50, 100, 200, 400, 800, 1600]


def tuned_params(feature_set: str, model: str = "LightGBM") -> dict:
    try:
        return M.read_json(f"metrics_{feature_set}.json")["models"][model]["best_params"]
    except FileNotFoundError:
        return {}


def fit_lgbm(params: dict, X_train, y_train, X_val, y_val):
    model = M.make_lgbm(**{**LGBM_FAST, **params})
    model.fit(X_train, y_train, eval_X=X_val, eval_y=y_val, eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    return model


def headline(y: np.ndarray, proba: np.ndarray) -> dict:
    scores = M.scores_from_confusion(M.confusion(y, proba.argmax(axis=1)))
    return {k: round(float(scores[k]), 4) for k in ("accuracy", "macro_f1", "qwk", "within_one")}


def ablation_variants() -> list[tuple[str, str, list[str]]]:
    macros, label = config.MACROS, config.FEATURE_SETS["label"]
    variants = [("base", "macros", macros)]
    variants += [("add_one", f"macros + {f}", macros + [f]) for f in config.LABEL_EXTRA]
    variants += [("base", "label", label)]
    variants += [("drop_one", f"label − {f}", [c for c in label if c != f]) for f in label]
    variants += [
        ("beyond_label", "label + engineered ratios", label + M.ENGINEERED),
        ("beyond_label", "label + food group", label + [config.GROUP_COLUMN]),
        ("beyond_label", "label + fruit/veg share", label + ["fvl"]),
        ("beyond_label", "label + food group + fruit/veg share", label + EXTRA),
    ]
    return variants


def frame() -> pd.DataFrame:
    df = M.load_dataset()
    return pd.concat([df, M.add_engineered(df)], axis=1)


def run_ablation() -> list[dict]:
    df = frame()
    params = tuned_params("label")
    rows = []
    for kind, name, features in ablation_variants():
        X_train, y_train = M.split_xy(df, "train", features)
        X_val, y_val = M.split_xy(df, "val", features)
        X_test, y_test = M.split_xy(df, "test", features)
        started = time.perf_counter()
        model = fit_lgbm(params, X_train, y_train, X_val, y_val)
        row = {"kind": kind, "variant": name, "n_features": len(features),
               "trees": int(model.best_iteration_), "seconds": round(time.perf_counter() - started, 1),
               **headline(y_test, model.predict_proba(X_test))}
        rows.append(row)
        print(f"[ablation] {name}: macro-F1 {row['macro_f1']}, accuracy {row['accuracy']}", flush=True)
        M.write_json("ablation.json", rows)
    return rows


def run_learning_curves() -> list[dict]:
    df = M.load_dataset()
    rng = np.random.default_rng(config.SEED)
    rows = []
    for feature_set, model_name in [("macros", "LightGBM"), ("label", "LightGBM"), ("label", "LogReg")]:
        features = config.FEATURE_SETS[feature_set]
        X_train, y_train = M.split_xy(df, "train", features)
        X_val, y_val = M.split_xy(df, "val", features)
        params = tuned_params(feature_set, model_name)
        order = rng.permutation(len(X_train))
        for size in [s for s in LEARNING_SIZES if s < len(X_train)] + [len(X_train)]:
            idx = order[:size]  # nested subsets: every larger sample contains the smaller ones
            X_part, y_part = X_train.iloc[idx], y_train[idx]
            if model_name == "LightGBM":
                fitted = fit_lgbm({**params, "min_child_samples": min(params.get("min_child_samples", 20),
                                                                      max(5, size // 100))},
                                  X_part, y_part, X_val, y_val)
            else:
                fitted = M.make_logreg(**params).fit(X_part, y_part)
            row = {"feature_set": feature_set, "model": model_name, "train_rows": size,
                   "train": headline(y_part, fitted.predict_proba(X_part)),
                   "val": headline(y_val, fitted.predict_proba(X_val))}
            rows.append(row)
            print(f"[learning] {model_name}/{feature_set} n={size}: val macro-F1 {row['val']['macro_f1']}",
                  flush=True)
            M.write_json("learning_curves.json", rows)
    return rows


def run_compact() -> list[dict]:
    """Accuracy, size and latency of the tuned LightGBM cut to its first k boosting rounds."""
    df = M.load_dataset()
    rows = []
    for feature_set, features in config.FEATURE_SETS.items():
        model = joblib.load(config.MODELS_DIR / f"{feature_set}_LightGBM.joblib")
        X_val, y_val = M.split_xy(df, "val", features)
        X_test, y_test = M.split_xy(df, "test", features)
        full = int(model.best_iteration_)
        singles = [X_test.iloc[[i]] for i in range(200)]
        for rounds in [k for k in COMPACT_ROUNDS if k < full] + [full]:
            started = time.perf_counter()
            for row in singles:
                model.predict_proba(row, num_iteration=rounds)
            single_ms = (time.perf_counter() - started) / len(singles) * 1e3
            text = model.booster_.model_to_string(num_iteration=rounds)
            row = {"feature_set": feature_set, "rounds": rounds, "is_full": rounds == full,
                   "size_mb": round(len(text.encode("utf-8")) / 1e6, 2), "single_ms": round(single_ms, 2),
                   "val": headline(y_val, model.predict_proba(X_val, num_iteration=rounds)),
                   "test": headline(y_test, model.predict_proba(X_test, num_iteration=rounds))}
            rows.append(row)
            print(f"[compact] {feature_set} {rounds} rounds: {row['size_mb']} MB, "
                  f"test accuracy {row['test']['accuracy']}, macro-F1 {row['test']['macro_f1']}", flush=True)
            M.write_json("compact_models.json", rows)
    return rows


if __name__ == "__main__":
    config.ensure_dirs()
    todo = sys.argv[1:] or ["ablation", "learning", "compact"]
    if "ablation" in todo:
        run_ablation()
    if "learning" in todo:
        run_learning_curves()
    if "compact" in todo:
        run_compact()
