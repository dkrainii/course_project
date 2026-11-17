"""Reference baseline: the published Nutri-Score algorithm (2023 revision, general foods),
computed only from what a nutrition label gives.

The real score also needs the share of fruit/vegetables/legumes and the product category
(beverages, fats/oils/nuts, cheese and red meat have their own rules). A label has neither, so
here every product is scored as a general food with 0 % fruit and vegetables. The gap between
this baseline and the trained models shows what the models recover from the nutrients alone.

    python -m ml.nutriscore.formula      # checks the implementation against Open Food Facts
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from . import modeling as M

KJ_PER_KCAL = 4.184
# A component earns one point for every threshold it exceeds.
ENERGY_KJ = np.arange(335, 3351, 335)
SUGARS = np.array([3.4, 6.8, 10, 14, 17, 20, 24, 27, 31, 34, 37, 41, 44, 48, 51])
SATURATED_FAT = np.arange(1, 11)
SALT = np.round(np.arange(0.2, 4.01, 0.2), 1)
PROTEINS = np.array([2.4, 4.8, 7.2, 9.6, 12, 14, 17])
FIBER = np.array([3.0, 4.1, 5.2, 6.3, 7.4])
GRADE_UPPER_BOUNDS = np.array([0, 2, 10, 18])  # A ≤ 0 < B ≤ 2 < C ≤ 10 < D ≤ 18 < E

SPECIAL_GROUPS = {"beverages", "fat and sauces", "milk and dairy products", "alcoholic beverages",
                  "fish meat eggs", "salty snacks"}  # contain categories with their own rules


def points(values: pd.Series, thresholds: np.ndarray) -> np.ndarray:
    return np.searchsorted(thresholds, values.fillna(0).to_numpy(), side="left")


def fvl_points(share: pd.Series) -> np.ndarray:
    s = share.fillna(0).to_numpy()
    return np.select([s > 80, s > 60, s > 40], [5, 2, 1], default=0)


def score(df: pd.DataFrame, use_fvl: bool = False, label_kj: bool = False) -> np.ndarray:
    """Nutri-Score points. `label_kj` takes the energy in kJ as printed on the label (what Open
    Food Facts scores); otherwise it is converted from kcal, the only energy value Kolos stores."""
    energy_kj = df["kcal"] * KJ_PER_KCAL
    if label_kj:
        energy_kj = df["energy_kj"].fillna(energy_kj)
    negative = (points(energy_kj, ENERGY_KJ) + points(df["sugars"], SUGARS)
                + points(df["saturated_fat"], SATURATED_FAT) + points(df["salt"], SALT))
    fibre_fvl = points(df["fiber"], FIBER) + (fvl_points(df["fvl"]) if use_fvl else 0)
    protein = points(df["proteins"], PROTEINS)
    # With 11+ negative points protein no longer offsets them.
    return np.where(negative >= 11, negative - fibre_fvl, negative - fibre_fvl - protein)


def grade_index(scores: np.ndarray) -> np.ndarray:
    return np.searchsorted(GRADE_UPPER_BOUNDS, scores, side="left")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    """One-hot 'probabilities', so the baseline goes through the same evaluation code."""
    return np.eye(M.N_CLASSES)[grade_index(score(df))]


def validate(df: pd.DataFrame) -> dict:
    """Agreement with the scores Open Food Facts publishes, where the simplification should not
    matter: general foods with a known fruit/vegetable share and all label fields present.

    Reported twice: with the label's own kJ (checks the implementation) and with kJ converted
    from kcal (what the baseline has to work with), plus the agreement per food subgroup.
    """
    general = (~df[config.GROUP_COLUMN].astype(object).isin(SPECIAL_GROUPS)
               & df[config.GROUP_COLUMN].notna() & df["fvl"].notna() & df["off_score"].notna()
               & df[config.LABEL_EXTRA].notna().all(axis=1))
    part = df[general]
    off, y = part["off_score"].to_numpy(), part["y"].to_numpy()

    def agreement(ours: np.ndarray) -> dict:
        diff = ours - off
        return {
            "score_exact_match": round(float((diff == 0).mean()), 4),
            "score_within_1": round(float((np.abs(diff) <= 1).mean()), 4),
            "grade_match": round(float((grade_index(ours) == y).mean()), 4),
        }

    with_label_kj = score(part, use_fvl=True, label_kj=True)
    diff = pd.Series(with_label_kj - off, index=part.index)
    by_subgroup = (diff.groupby(part["food_subgroup"])
                   .agg(rows="size", exact=lambda d: (d == 0).mean(), mean_diff="mean")
                   .sort_values("rows", ascending=False))
    on_salt_threshold = np.isclose(part["salt"].to_numpy() / 0.2 % 1, 0, atol=1e-6) & (part["salt"] > 0).to_numpy()
    return {
        "rows_checked": int(len(part)),
        "label_kj": agreement(with_label_kj),
        "kj_from_kcal": agreement(score(part, use_fvl=True)),
        "energy_points_changed_by_conversion": round(float(
            (points(part["kcal"] * KJ_PER_KCAL, ENERGY_KJ)
             != points(part["energy_kj"].fillna(part["kcal"] * KJ_PER_KCAL), ENERGY_KJ)).mean()), 4),
        "score_diff_distribution_label_kj": {
            str(int(k)): int(v) for k, v in diff.clip(-5, 5).value_counts().sort_index().items()},
        "salt_exactly_on_threshold": {
            "share_of_rows": round(float(on_salt_threshold.mean()), 4),
            "exact_on_threshold": round(float((diff[on_salt_threshold] == 0).mean()), 4),
            "exact_elsewhere": round(float((diff[~on_salt_threshold] == 0).mean()), 4),
        },
        "by_subgroup_label_kj": {
            str(name): {"rows": int(r.rows), "exact": round(float(r.exact), 4), "mean_diff": round(float(r.mean_diff), 3)}
            for name, r in by_subgroup[by_subgroup["rows"] >= 300].iterrows()},
    }


def run() -> dict:
    df = M.load_dataset()
    test = df[df["split"] == "test"]
    y = test["y"].to_numpy()
    proba = predict_proba(test)
    result = {
        "validation_against_off": validate(df),
        "test": M.evaluate(y, (proba + 1e-6) / (1 + 1e-6 * M.N_CLASSES)),
        "test_ci95": M.bootstrap_ci(y, proba.argmax(axis=1)),
        "test_by_group": {
            str(group): round(float((proba[mask].argmax(axis=1) == y[mask]).mean()), 4)
            for group in test[config.GROUP_COLUMN].cat.categories
            if (mask := (test[config.GROUP_COLUMN] == group).to_numpy()).sum() >= 30
        },
    }
    result["test"].pop("log_loss")  # not meaningful for hard predictions
    M.write_json("formula_baseline.json", result)
    return result


if __name__ == "__main__":
    import json

    out = run()
    print(json.dumps({k: v for k, v in out.items() if k != "test"} | {
        "test": {k: v for k, v in out["test"].items() if k not in ("per_class", "confusion")}}, indent=2))
