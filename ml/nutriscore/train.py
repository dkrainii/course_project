"""Step 4. Tune, train and compare the models on one feature set.

Protocol (identical for every model, so the comparison is fair):
  1. hyper-parameters — search with 3-fold stratified CV on a subsample of the training part,
     scored by macro-F1;
  2. the best configuration is refitted on the whole training part (LightGBM uses the validation
     part for early stopping);
  3. the validation part is for choosing between models, the test part is touched once, for the
     numbers that go into the report (with bootstrap confidence intervals and McNemar's test).

    python -m ml.nutriscore.train --features macros label
"""
from __future__ import annotations

import argparse
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, ParameterSampler, StratifiedKFold

from . import config
from . import modeling as M

TUNE_ROWS = 200_000
CV_FOLDS = 3

SEARCH_SPACES = {
    "Dummy": [{}],
    "LogReg": list(ParameterGrid({"C": [0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]})),
    "RandomForest": {
        "max_depth": [10, 14, 18, 22],
        "min_samples_leaf": [5, 10, 20, 50],
        "max_features": ["sqrt", 0.6, 1.0],
        "class_weight": [None, "balanced_subsample"],
    },
    "LightGBM": {
        "num_leaves": [31, 63, 127, 255],
        "min_child_samples": [20, 50, 100, 200],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.0, 1.0, 5.0, 10.0],
        "class_weight": [None, "balanced"],
    },
}
RANDOM_TRIALS = {"RandomForest": 10, "LightGBM": 15}
# Cheaper settings while searching; the final fit uses FINAL_PARAMS.
SEARCH_PARAMS = {
    "RandomForest": {"n_estimators": 100},
    "LightGBM": {"n_estimators": 300, "learning_rate": 0.1, "subsample_freq": 1},
}
FINAL_PARAMS = {
    "RandomForest": {"n_estimators": 200, "max_samples": 0.5},
    "LightGBM": {"n_estimators": 4000, "learning_rate": 0.05, "subsample_freq": 1},
}
EARLY_STOPPING_ROUNDS = 100


def candidates(name: str, quick: bool) -> list[dict]:
    space = SEARCH_SPACES[name]
    if isinstance(space, list):
        return space[:2] if quick else space
    n = 2 if quick else RANDOM_TRIALS[name]
    return list(ParameterSampler(space, n_iter=n, random_state=config.SEED))


def cross_validate(name: str, params: dict, X: pd.DataFrame, y: np.ndarray) -> dict:
    folds = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=config.SEED)
    f1, acc = [], []
    started = time.perf_counter()
    for train_idx, valid_idx in folds.split(X, y):
        model = M.FACTORIES[name](**{**SEARCH_PARAMS.get(name, {}), **params})
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[valid_idx])
        f1.append(f1_score(y[valid_idx], pred, average="macro"))
        acc.append(float((pred == y[valid_idx]).mean()))
    return {
        "params": params,
        "cv_macro_f1": round(float(np.mean(f1)), 4),
        "cv_macro_f1_std": round(float(np.std(f1)), 4),
        "cv_accuracy": round(float(np.mean(acc)), 4),
        "seconds": round(time.perf_counter() - started, 1),
    }


def fit_final(name: str, params: dict, X_train, y_train, X_val, y_val):
    model = M.FACTORIES[name](**{**FINAL_PARAMS.get(name, {}), **params})
    started = time.perf_counter()
    if name == "LightGBM":
        model.fit(X_train, y_train, eval_X=X_val, eval_y=y_val, eval_metric="multi_logloss",
                  callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    else:
        model.fit(X_train, y_train)
    return model, round(time.perf_counter() - started, 1)


def run(feature_set: str, quick: bool = False) -> dict:
    config.ensure_dirs()
    features = config.FEATURE_SETS[feature_set]
    df = M.load_dataset()
    X_train, y_train = M.split_xy(df, "train", features)
    X_val, y_val = M.split_xy(df, "val", features)
    X_test, y_test = M.split_xy(df, "test", features)

    rng = np.random.default_rng(config.SEED)
    tune_idx = rng.choice(len(X_train), min(len(X_train), TUNE_ROWS), replace=False)
    X_tune, y_tune = X_train.iloc[tune_idx], y_train[tune_idx]

    results: dict = {
        "feature_set": feature_set, "features": features,
        "rows": {"train": len(X_train), "val": len(X_val), "test": len(X_test), "tuning": len(X_tune)},
        "protocol": {"cv_folds": CV_FOLDS, "search_params": SEARCH_PARAMS, "final_params": FINAL_PARAMS,
                     "early_stopping_rounds": EARLY_STOPPING_ROUNDS},
        "models": {},
    }
    test_proba: dict[str, np.ndarray] = {}
    val_proba: dict[str, np.ndarray] = {}

    for name in M.MODEL_ORDER:
        print(f"[{feature_set}] {name}: tuning", flush=True)
        trials = [cross_validate(name, params, X_tune, y_tune) for params in candidates(name, quick)]
        best = max(trials, key=lambda t: t["cv_macro_f1"])
        print(f"[{feature_set}] {name}: best CV macro-F1 {best['cv_macro_f1']} with {best['params']}", flush=True)

        model, fit_seconds = fit_final(name, best["params"], X_train, y_train, X_val, y_val)
        val_proba[name] = model.predict_proba(X_val)
        test_proba[name] = model.predict_proba(X_test)
        entry = {
            "best_params": best["params"],
            "cv_macro_f1": best["cv_macro_f1"],
            "cv_macro_f1_std": best["cv_macro_f1_std"],
            "trials": trials,
            "fit_seconds": fit_seconds,
            "val": M.evaluate(y_val, val_proba[name]),
            "test": M.evaluate(y_test, test_proba[name]),
            "test_ci95": M.bootstrap_ci(y_test, test_proba[name].argmax(axis=1)),
            "latency": M.latency(model, X_test),
            "size_mb": M.model_size_mb(model),
        }
        if name == "LightGBM":
            entry["best_iteration"] = int(model.best_iteration_)
        results["models"][name] = entry
        joblib.dump(model, config.MODELS_DIR / f"{feature_set}_{name}.joblib", compress=3)
        print(f"[{feature_set}] {name}: test accuracy {entry['test']['accuracy']}, "
              f"macro-F1 {entry['test']['macro_f1']}, QWK {entry['test']['qwk']} "
              f"(fit {fit_seconds}s)", flush=True)

    preds = {name: p.argmax(axis=1) for name, p in test_proba.items()}
    pairs = [("LogReg", "Dummy"), ("RandomForest", "LogReg"), ("LightGBM", "RandomForest"), ("LightGBM", "LogReg")]
    results["mcnemar"] = {f"{a} vs {b}": M.mcnemar(y_test, preds[a], preds[b]) for a, b in pairs}

    np.savez_compressed(
        config.PROCESSED_DIR / f"predictions_{feature_set}.npz", y_test=y_test, y_val=y_val,
        **{f"test_{name}": p.astype(np.float32) for name, p in test_proba.items()},
        **{f"val_{name}": p.astype(np.float32) for name, p in val_proba.items()},
    )
    M.write_json(f"metrics_{feature_set}.json", results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--features", nargs="+", default=list(config.FEATURE_SETS), choices=list(config.FEATURE_SETS))
    parser.add_argument("--quick", action="store_true", help="two trials per model (smoke test)")
    args = parser.parse_args()
    for feature_set in args.features:
        run(feature_set, quick=args.quick)
