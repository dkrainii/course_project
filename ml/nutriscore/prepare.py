"""Step 2. Graded Open Food Facts rows → clean modelling dataset with a fixed train/val/test split.

Every cleaning rule is counted, so the report can show how many rows each one removed
(results/cleaning_funnel.json).

    python -m ml.nutriscore.prepare
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

NUTRIENTS = config.MACROS + config.LABEL_EXTRA
KJ_PER_KCAL = 4.184
SALT_PER_SODIUM = 2.5


def harmonise(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw OFF columns → one tidy column per quantity (still uncleaned)."""
    kcal = raw["energy-kcal_100g"]
    kcal = kcal.fillna(raw["energy-kj_100g"] / KJ_PER_KCAL).fillna(raw["energy_100g"] / KJ_PER_KCAL)
    group = raw["pnns_groups_1"].str.strip().str.lower().str.replace("-", " ")
    group = group.where(group.notna() & (group != "unknown"))
    countries = raw["countries_tags"].fillna("")
    return pd.DataFrame({
        "code": raw["code"],
        "product_name": raw["product_name"],
        "grade": raw["nutriscore_grade"],
        "off_score": raw["nutriscore_score"],
        "kcal": kcal,
        "energy_kj": raw["energy-kj_100g"].fillna(raw["energy_100g"]),  # as printed on the label
        "proteins": raw["proteins_100g"],
        "fats": raw["fat_100g"],
        "carbs": raw["carbohydrates_100g"],
        "sugars": raw["sugars_100g"],
        "saturated_fat": raw["saturated-fat_100g"],
        "fiber": raw["fiber_100g"],
        "salt": raw["salt_100g"].fillna(raw["sodium_100g"] * SALT_PER_SODIUM),
        "alcohol": raw["alcohol_100g"],
        "fvl": raw["fruits-vegetables-legumes_100g"],
        config.GROUP_COLUMN: group,
        "food_subgroup": raw["pnns_groups_2"],
        "nova_group": raw["nova_group"],
        "is_ukraine": countries.str.contains("en:ukraine", regex=False),
        "n_countries": countries.str.count(",") + (countries != "").astype(int),
        "created_year": pd.to_datetime(raw["created_t"], unit="s", errors="coerce").dt.year,
        "unique_scans_n": raw["unique_scans_n"],
        "has_off_errors": raw["data_quality_errors_tags"].notna(),
    })


def atwater_kcal(df: pd.DataFrame) -> pd.Series:
    """Energy implied by the macronutrients (4/4/9 kcal per g, 7 for alcohol)."""
    return 4 * df["proteins"] + 4 * df["carbs"] + 9 * df["fats"] + 7 * df["alcohol"].fillna(0)


def energy_consistent(df: pd.DataFrame) -> pd.Series:
    estimate = atwater_kcal(df)
    tolerance = np.maximum(config.ENERGY_ABS_TOL, config.ENERGY_REL_TOL * estimate)
    return (df["kcal"] - estimate).abs() <= tolerance


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    funnel: list[dict] = []

    def step(name: str, mask: pd.Series | None = None) -> None:
        nonlocal df
        before = len(df)
        if mask is not None:
            df = df[mask.to_numpy()]
        funnel.append({"step": name, "rows": len(df), "removed": before - len(df)})

    step("graded a–e in the dump")
    step("energy and the three macronutrients present", df[config.MACROS].notna().all(axis=1))
    step("no Open Food Facts data-quality errors", ~df["has_off_errors"])

    grams = [c for c in NUTRIENTS if c != "kcal"]
    in_range = df[grams].isna() | ((df[grams] >= 0) & (df[grams] <= 100))
    step("values in the physical range (0–100 g, 0–950 kcal)",
         in_range.all(axis=1) & df["kcal"].between(0, config.MAX_KCAL))
    step("proteins + fats + carbs ≤ 105 g",
         df[["proteins", "fats", "carbs"]].sum(axis=1) <= config.MAX_MACRO_SUM)
    step("sugars ≤ carbs and saturated fat ≤ fats",
         ~(df["sugars"] > df["carbs"] + 1) & ~(df["saturated_fat"] > df["fats"] + 1))
    step("label energy agrees with the macronutrients (Atwater)", energy_consistent(df))

    df = df.copy()
    df[NUTRIENTS] = df[NUTRIENTS].round(2)
    step("exact duplicates (same nutrients and grade) removed",
         ~df.duplicated(subset=NUTRIENTS + ["grade"], keep="first"))
    return df.reset_index(drop=True), funnel


def assign_split(df: pd.DataFrame) -> pd.Series:
    """Deterministic split by a hash of the nutrient profile.

    Products with an identical profile always land in the same part, so the test set never
    contains a nutrient vector the model has already seen during training.
    """
    keys = df[NUTRIENTS].fillna(-1.0)
    hashed = pd.util.hash_pandas_object(keys, index=False, hash_key=f"{config.SEED:016d}")
    u = (hashed.to_numpy() % 10_000) / 10_000
    split = np.where(u < config.TEST_SIZE, "test",
                     np.where(u < config.TEST_SIZE + config.VAL_SIZE, "val", "train"))
    return pd.Series(split, index=df.index, name="split")


def prepare(src: Path = config.EXTRACTED, dst: Path = config.DATASET) -> dict:
    config.ensure_dirs()
    tidy = harmonise(pd.read_parquet(src))
    df, funnel = clean(tidy)
    df["y"] = df["grade"].map({g: i for i, g in enumerate(config.GRADES)}).astype("int8")
    df["split"] = assign_split(df)
    df = df.drop(columns=["has_off_errors"])
    df.to_parquet(dst, index=False, compression="zstd")

    by_split = (df.groupby("split")["grade"].value_counts().unstack(fill_value=0)
                .reindex(["train", "val", "test"]))
    summary = {
        "funnel": funnel,
        "rows": len(df),
        "splits": {name: {"rows": int(row.sum()), **{g: int(row[g]) for g in config.GRADES}}
                   for name, row in by_split.iterrows()},
        "missing_share": {c: round(float(df[c].isna().mean()), 4)
                          for c in config.LABEL_EXTRA + ["fvl", config.GROUP_COLUMN]},
        "ukraine_rows": int(df["is_ukraine"].sum()),
        "thresholds": {
            "max_kcal": config.MAX_KCAL, "max_macro_sum": config.MAX_MACRO_SUM,
            "energy_abs_tol": config.ENERGY_ABS_TOL, "energy_rel_tol": config.ENERGY_REL_TOL,
        },
    }
    (config.RESULTS_DIR / "cleaning_funnel.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2, ensure_ascii=False))
