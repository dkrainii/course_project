"""Shared settings of the Nutri-Score experiment: paths, columns, feature sets, random seed."""
from __future__ import annotations

from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ML_ROOT / "data"
RAW_DUMP = DATA_DIR / "raw" / "en.openfoodfacts.org.products.csv.gz"
PROCESSED_DIR = DATA_DIR / "processed"
EXTRACTED = PROCESSED_DIR / "off_graded.parquet"       # graded rows, raw values
DATASET = PROCESSED_DIR / "nutriscore_dataset.parquet"  # cleaned, with the train/val/test split
MODELS_DIR = ML_ROOT / "models" / "nutriscore"

REPORT_DIR = ML_ROOT / "reports" / "nutriscore"
FIGURES_DIR = REPORT_DIR / "figures"
RESULTS_DIR = REPORT_DIR / "results"  # JSON / CSV with every number quoted in the report

SEED = 42
GRADES = ("a", "b", "c", "d", "e")

# --- columns of the Open Food Facts CSV export that the experiment reads
TEXT_COLUMNS = [
    "code", "product_name", "countries_tags", "nutriscore_grade", "pnns_groups_1", "pnns_groups_2",
    "food_groups_tags", "main_category", "data_quality_errors_tags",
]
NUMERIC_COLUMNS = [
    "nutriscore_score", "nova_group", "created_t", "completeness", "unique_scans_n",
    "energy-kcal_100g", "energy-kj_100g", "energy_100g",
    "fat_100g", "saturated-fat_100g", "carbohydrates_100g", "sugars_100g", "fiber_100g",
    "proteins_100g", "salt_100g", "sodium_100g", "alcohol_100g", "fruits-vegetables-legumes_100g",
]
RAW_COLUMNS = TEXT_COLUMNS + NUMERIC_COLUMNS

# --- model inputs (all per 100 g)
MACROS = ["kcal", "proteins", "fats", "carbs"]                  # what every Kolos product has
LABEL_EXTRA = ["sugars", "saturated_fat", "fiber", "salt"]      # the rest of a nutrition label
FEATURE_SETS: dict[str, list[str]] = {
    "macros": MACROS,
    "label": MACROS + LABEL_EXTRA,
}
GROUP_COLUMN = "food_group"  # coarse food category (PNNS group 1), used only in the ablation

# --- cleaning thresholds (see reports/nutriscore/README.md, "Очищення даних")
MAX_KCAL = 950.0            # same sanity limit as food_db.parse_off_product
MAX_MACRO_SUM = 105.0       # proteins + fats + carbs per 100 g, with a rounding allowance
ENERGY_ABS_TOL = 50.0       # |label kcal − Atwater kcal| allowed: max(abs, rel · Atwater)
ENERGY_REL_TOL = 0.25

TEST_SIZE = 0.15
VAL_SIZE = 0.15


def ensure_dirs() -> None:
    for path in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
