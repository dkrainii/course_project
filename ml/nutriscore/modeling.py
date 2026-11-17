"""Building blocks shared by the training scripts: data access, model factories, metrics."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config

N_CLASSES = len(config.GRADES)
MODEL_ORDER = ["Dummy", "LogReg", "RandomForest", "LightGBM"]


# --------------------------------------------------------------------------- data

def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(config.DATASET)
    df[config.GROUP_COLUMN] = df[config.GROUP_COLUMN].astype("category")
    return df


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios a nutritionist would look at; trees can only approximate them from raw columns."""
    kcal = df["kcal"].clip(lower=1)
    out = pd.DataFrame(index=df.index)
    out["protein_energy_share"] = (4 * df["proteins"] / kcal).clip(0, 1.5)
    out["fat_energy_share"] = (9 * df["fats"] / kcal).clip(0, 1.5)
    out["sugar_share_of_carbs"] = (df["sugars"] / df["carbs"].clip(lower=0.5)).clip(0, 1.5)
    out["saturated_share_of_fat"] = (df["saturated_fat"] / df["fats"].clip(lower=0.5)).clip(0, 1.5)
    out["non_macro_mass"] = 100 - df["proteins"] - df["fats"] - df["carbs"]  # ≈ water + fibre + ash
    return out


ENGINEERED = ["protein_energy_share", "fat_energy_share", "sugar_share_of_carbs",
              "saturated_share_of_fat", "non_macro_mass"]


def split_xy(df: pd.DataFrame, split: str, features: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    part = df[df["split"] == split]
    return part[features], part["y"].to_numpy()


# --------------------------------------------------------------------------- models

def make_dummy(**params):
    return DummyClassifier(strategy="prior")


def make_logreg(C: float = 1.0, class_weight=None, **_):
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=C, class_weight=class_weight, max_iter=500),
    )


def make_forest(n_estimators: int = 200, n_jobs: int = -1, **params):
    return RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs, random_state=config.SEED, **params)


def make_lgbm(n_estimators: int = 400, n_jobs: int = -1, **params):
    return LGBMClassifier(objective="multiclass", n_estimators=n_estimators, n_jobs=n_jobs,
                          random_state=config.SEED, verbose=-1, **params)


FACTORIES = {"Dummy": make_dummy, "LogReg": make_logreg, "RandomForest": make_forest, "LightGBM": make_lgbm}


# --------------------------------------------------------------------------- metrics

def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.bincount(y_true * N_CLASSES + y_pred, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)


def scores_from_confusion(cm: np.ndarray) -> dict[str, float]:
    """All headline metrics from a confusion matrix (rows = true grade, columns = predicted)."""
    total = cm.sum()
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    recall = np.divide(tp, support, out=np.zeros(N_CLASSES), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(N_CLASSES), where=predicted > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(N_CLASSES), where=denom > 0)

    idx = np.arange(N_CLASSES)
    distance = np.abs(idx[:, None] - idx[None, :])
    expected = np.outer(support, predicted) / total
    weights = distance ** 2
    kappa = 1 - (weights * cm).sum() / (weights * expected).sum() if (weights * expected).sum() else 0.0
    return {
        "accuracy": tp.sum() / total,
        "balanced_accuracy": recall.mean(),
        "macro_f1": f1.mean(),
        "weighted_f1": (f1 * support).sum() / total,
        "qwk": kappa,                                   # quadratic weighted kappa: grades are ordered
        "mae_grades": (distance * cm).sum() / total,    # mean error in grade steps
        "within_one": cm[distance <= 1].sum() / total,  # exact or adjacent grade
    }


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    y_pred = proba.argmax(axis=1)
    cm = confusion(y_true, y_pred)
    result = {k: round(float(v), 4) for k, v in scores_from_confusion(cm).items()}
    result["log_loss"] = round(float(log_loss(y_true, proba, labels=list(range(N_CLASSES)))), 4)
    tp = np.diag(cm)
    result["per_class"] = {
        g: {
            "precision": round(float(tp[i] / cm[:, i].sum()) if cm[:, i].sum() else 0.0, 4),
            "recall": round(float(tp[i] / cm[i].sum()) if cm[i].sum() else 0.0, 4),
            "support": int(cm[i].sum()),
        }
        for i, g in enumerate(config.GRADES)
    }
    for stats in result["per_class"].values():
        p, r = stats["precision"], stats["recall"]
        stats["f1"] = round(2 * p * r / (p + r), 4) if p + r else 0.0
    result["confusion"] = cm.tolist()
    return result


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000,
                 keys: tuple[str, ...] = ("accuracy", "macro_f1", "qwk")) -> dict[str, list[float]]:
    """95 % percentile intervals of the test metrics (resampling test products with replacement)."""
    rng = np.random.default_rng(config.SEED)
    cell = y_true * N_CLASSES + y_pred
    samples = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        picked = cell[rng.integers(0, len(cell), len(cell))]
        cm = np.bincount(picked, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
        scores = scores_from_confusion(cm)
        for k in keys:
            samples[k][b] = scores[k]
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
            for k, v in samples.items()}


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """McNemar's test (with continuity correction) on the products two models disagree about."""
    from scipy.stats import chi2

    a_right, b_right = pred_a == y_true, pred_b == y_true
    only_a, only_b = int((a_right & ~b_right).sum()), int((~a_right & b_right).sum())
    stat = (abs(only_a - only_b) - 1) ** 2 / (only_a + only_b) if only_a + only_b else 0.0
    return {"only_first_right": only_a, "only_second_right": only_b,
            "statistic": round(float(stat), 2), "p_value": float(chi2.sf(stat, 1))}


# --------------------------------------------------------------------------- costs

def model_size_mb(model) -> float:
    path = config.MODELS_DIR / "_size_probe.joblib"
    joblib.dump(model, path, compress=3)
    size = path.stat().st_size / 1e6
    path.unlink()
    return round(size, 2)


def latency(model, X: pd.DataFrame, single_calls: int = 200) -> dict[str, float]:
    """Prediction cost: per row in a big batch, and for one product at a time (the web app's case)."""
    batch = X.iloc[: min(len(X), 20_000)]
    started = time.perf_counter()
    model.predict_proba(batch)
    per_row_us = (time.perf_counter() - started) / len(batch) * 1e6
    rows = [X.iloc[[i]] for i in range(min(single_calls, len(X)))]
    started = time.perf_counter()
    for row in rows:
        model.predict_proba(row)
    single_ms = (time.perf_counter() - started) / len(rows) * 1e3
    return {"batch_us_per_row": round(per_row_us, 2), "single_ms": round(single_ms, 3)}


def write_json(name: str, payload) -> Path:
    path = config.RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_json(name: str):
    return json.loads((config.RESULTS_DIR / name).read_text(encoding="utf-8"))
