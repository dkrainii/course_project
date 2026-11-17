"""Step 3. Exploratory analysis of the cleaned dataset: figures 01–08 and results/eda_stats.json.

    python -m ml.nutriscore.eda
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from .. import plotting as P
from . import config
from .prepare import NUTRIENTS, atwater_kcal, harmonise

FEATURE_LABELS = {
    "kcal": "Енергія, ккал",
    "proteins": "Білки, г",
    "fats": "Жири, г",
    "carbs": "Вуглеводи, г",
    "sugars": "Цукри, г",
    "saturated_fat": "Насичені жири, г",
    "fiber": "Клітковина, г",
    "salt": "Сіль, г",
    "fvl": "Фрукти/овочі/бобові, %",
    "food_group": "Група продуктів",
}
GROUP_LABELS = {
    "sugary snacks": "Солодкі снеки",
    "fish meat eggs": "Риба, м'ясо, яйця",
    "milk and dairy products": "Молочні продукти",
    "cereals and potatoes": "Зернові та картопля",
    "fat and sauces": "Жири та соуси",
    "fruits and vegetables": "Фрукти та овочі",
    "composite foods": "Готові страви",
    "beverages": "Напої",
    "salty snacks": "Солоні снеки",
    "alcoholic beverages": "Алкогольні напої",
}
FUNNEL_LABELS = {
    "graded a–e in the dump": "Є оцінка Nutri-Score (A–E)",
    "energy and the three macronutrients present": "Є енергія, білки, жири, вуглеводи",
    "no Open Food Facts data-quality errors": "Без позначок помилок OFF",
    "values in the physical range (0–100 g, 0–950 kcal)": "Значення у фізичних межах",
    "proteins + fats + carbs ≤ 105 g": "Б + Ж + В ≤ 105 г",
    "sugars ≤ carbs and saturated fat ≤ fats": "Цукри ≤ вуглеводи, насичені ≤ жири",
    "label energy agrees with the macronutrients (Atwater)": "Енергія узгоджена з макронутрієнтами",
    "exact duplicates (same nutrients and grade) removed": "Без точних дублікатів",
}
GRADE_NAMES = [g.upper() for g in config.GRADES]


def fig_funnel(funnel: list[dict]):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    labels = [FUNNEL_LABELS.get(s["step"], s["step"]) for s in funnel][::-1]
    rows = [s["rows"] for s in funnel][::-1]
    ax.barh(labels, rows, height=0.55, color=P.BLUE)
    for y, (value, s) in enumerate(zip(rows, funnel[::-1])):
        note = P.thousands(value) + (f"  (−{P.thousands(s['removed'])})" if s["removed"] else "")
        ax.text(value + max(rows) * 0.01, y, note, va="center", fontsize=9, color=P.INK_SECONDARY)
    ax.set_xlim(0, max(rows) * 1.22)
    ax.xaxis.set_major_formatter(lambda v, _: P.thousands(v))
    ax.set_xlabel("Кількість продуктів")
    ax.set_title("Очищення даних: скільки записів лишається після кожного правила")
    P.horizontal_grid(ax)
    return fig


def fig_class_balance(df: pd.DataFrame):
    counts = df["grade"].value_counts().reindex(config.GRADES)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(GRADE_NAMES, counts.values, width=0.5, color=[P.GRADE_COLORS[g] for g in config.GRADES])
    for x, value in enumerate(counts.values):
        ax.text(x, value, f"{value / counts.sum():.1%}", ha="center", va="bottom",
                fontsize=9.5, color=P.INK_SECONDARY)
    ax.yaxis.set_major_formatter(lambda v, _: P.thousands(v))
    ax.set_xlabel("Оцінка Nutri-Score")
    ax.set_ylabel("Кількість продуктів")
    ax.set_title("Розподіл класів у очищеному наборі")
    return fig


def fig_energy_consistency(tidy: pd.DataFrame):
    """Label energy against the energy implied by the macros, before the consistency filter."""
    ok = tidy[config.MACROS].notna().all(axis=1) & tidy["kcal"].between(0, config.MAX_KCAL) \
        & tidy[["proteins", "fats", "carbs"]].apply(lambda s: s.between(0, 100)).all(axis=1)
    part = tidy[ok]
    part = part.sample(min(len(part), 300_000), random_state=config.SEED)
    estimate = atwater_kcal(part)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    hb = ax.hexbin(estimate, part["kcal"], gridsize=70, bins="log", cmap=P.SEQUENTIAL,
                   extent=(0, 950, 0, 950), mincnt=1, linewidths=0)
    x = np.linspace(0, 950, 200)
    tol = np.maximum(config.ENERGY_ABS_TOL, config.ENERGY_REL_TOL * x)
    for bound in (x - tol, x + tol):
        ax.plot(x, bound, color=P.ORANGE, linewidth=1.2)
    ax.text(600, 600 + 0.25 * 600 + 25, "межа допуску", color=P.INK_SECONDARY, fontsize=9, rotation=42,
            ha="center", va="bottom")
    ax.set_xlim(0, 950)
    ax.set_ylim(0, 950)
    ax.set_xlabel("Енергія за макронутрієнтами (4·Б + 4·В + 9·Ж), ккал")
    ax.set_ylabel("Енергія на етикетці, ккал")
    ax.set_title("Узгодженість енергії з макронутрієнтами")
    ax.grid(axis="x")
    fig.colorbar(hb, ax=ax, label="Кількість продуктів (лог. шкала)", shrink=0.8)
    return fig


def fig_features_by_grade(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.4))
    for ax, feature in zip(axes.ravel(), NUTRIENTS):
        data = [df.loc[df["grade"] == g, feature].dropna().to_numpy() for g in config.GRADES]
        box = ax.boxplot(data, tick_labels=GRADE_NAMES, showfliers=False, widths=0.55, patch_artist=True,
                         medianprops={"color": P.INK, "linewidth": 1.4},
                         whiskerprops={"color": P.MUTED, "linewidth": 1},
                         capprops={"color": P.MUTED, "linewidth": 1})
        for patch, g in zip(box["boxes"], config.GRADES):
            patch.set_facecolor(P.GRADE_COLORS[g])
            patch.set_edgecolor(P.SURFACE)
        ax.set_title(FEATURE_LABELS[feature], fontsize=10)
        ax.set_ylim(bottom=0)
    fig.suptitle("Нутрієнти на 100 г за оцінкою Nutri-Score (медіана, квартилі, 1,5·IQR)",
                 x=0.01, ha="left", fontsize=11.5, fontweight="semibold")
    fig.tight_layout()
    return fig


def fig_missing(df: pd.DataFrame):
    cols = config.LABEL_EXTRA + ["fvl", config.GROUP_COLUMN]
    share = df[cols].isna().mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh([FEATURE_LABELS[c] for c in share.index], share.values * 100, height=0.5, color=P.BLUE)
    for y, value in enumerate(share.values):
        ax.text(value * 100 + 1, y, f"{value:.1%}", va="center", fontsize=9.5, color=P.INK_SECONDARY)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Частка продуктів без значення, %")
    ax.set_title("Пропуски в додаткових ознаках")
    P.horizontal_grid(ax)
    return fig


def fig_correlation(df: pd.DataFrame):
    corr = df[NUTRIENTS + ["y"]].corr(method="spearman")
    labels = [FEATURE_LABELS[c].split(",")[0] for c in NUTRIENTS] + ["Оцінка (A→E)"]
    fig, ax = plt.subplots(figsize=(7, 5.8))
    im = ax.imshow(corr.values, cmap=P.DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.values[i, j]
            ax.text(j, i, f"{value:.2f}".replace("-", "−"), ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if abs(value) > 0.6 else P.INK)
    ax.grid(False)
    ax.spines[:].set_visible(False)
    ax.set_title("Кореляція Спірмена між ознаками та оцінкою")
    fig.colorbar(im, ax=ax, shrink=0.8, label="ρ Спірмена")
    return fig, corr


def fig_grade_by_group(df: pd.DataFrame):
    table = pd.crosstab(df[config.GROUP_COLUMN], df["grade"], normalize="index").reindex(columns=config.GRADES)
    counts = df[config.GROUP_COLUMN].value_counts()
    table = table.loc[(table["a"] + table["b"]).sort_values().index]
    names = [f"{GROUP_LABELS.get(g, g)}  ({P.thousands(counts[g])})" for g in table.index]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    left = np.zeros(len(table))
    for g in config.GRADES:
        width = table[g].to_numpy() * 100
        ax.barh(names, width, left=left, height=0.6, color=P.GRADE_COLORS[g], label=g.upper(),
                edgecolor=P.SURFACE, linewidth=1.5)
        for y, (start, w) in enumerate(zip(left, width)):
            if w >= 7:
                ax.text(start + w / 2, y, f"{w:.0f}", ha="center", va="center", fontsize=8.5,
                        color=P.GRADE_TEXT_ON_FILL[g])
        left += width
    ax.set_xlim(0, 100)
    ax.set_xlabel("Частка продуктів групи, %")
    ax.set_title("Оцінки Nutri-Score за групами продуктів (PNNS)", pad=30)
    ax.legend(ncol=5, loc="lower left", bbox_to_anchor=(-0.01, 0.99), handlelength=1, columnspacing=1.4)
    P.horizontal_grid(ax)
    ax.grid(False)
    return fig, table


def fig_macro_space(df: pd.DataFrame):
    """Mean grade over two pairs of macros: where four macros separate the grades and where not."""
    pairs = [("carbs", "fats"), ("proteins", "kcal")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, (x, y) in zip(axes, pairs):
        hb = ax.hexbin(df[x], df[y], C=df["y"], reduce_C_function=np.mean, gridsize=45,
                       cmap=P.DIVERGING, vmin=0, vmax=4, mincnt=15, linewidths=0)
        ax.set_xlabel(FEATURE_LABELS[x])
        ax.set_ylabel(FEATURE_LABELS[y])
        ax.grid(axis="x")
    cbar = fig.colorbar(hb, ax=axes, shrink=0.85, ticks=range(5))
    cbar.ax.set_yticklabels(GRADE_NAMES)
    cbar.set_label("Середня оцінка в комірці")
    fig.suptitle("Середня оцінка Nutri-Score у просторі макронутрієнтів (комірки з ≥ 15 продуктами)",
                 x=0.01, ha="left", fontsize=11.5, fontweight="semibold")
    return fig


def label_ambiguity(df: pd.DataFrame, features: list[str]) -> dict:
    """How often identical feature vectors carry different grades (a ceiling for any model)."""
    keys = df[features].fillna(-1.0)
    group = keys.groupby(features, sort=False).ngroup()
    per_group = df.groupby(group)["y"].agg(["size", "nunique"])
    majority = df.groupby([group, df["y"]]).size().groupby(level=0).max()
    repeated = per_group["size"] > 1
    return {
        "rows_in_repeated_profiles": int(per_group.loc[repeated, "size"].sum()),
        "share_rows_in_repeated_profiles": round(float(per_group.loc[repeated, "size"].sum() / len(df)), 4),
        "share_repeated_profiles_with_conflict": round(float((per_group.loc[repeated, "nunique"] > 1).mean()), 4),
        "memorisation_accuracy_ceiling": round(float(majority.sum() / len(df)), 4),
    }


def run() -> dict:
    config.ensure_dirs()
    P.apply_style()
    df = pd.read_parquet(config.DATASET)
    funnel = json.loads((config.RESULTS_DIR / "cleaning_funnel.json").read_text(encoding="utf-8"))["funnel"]
    tidy = harmonise(pd.read_parquet(config.EXTRACTED))

    P.save(fig_funnel(funnel), config.FIGURES_DIR / "01_cleaning_funnel.png")
    P.save(fig_energy_consistency(tidy), config.FIGURES_DIR / "02_energy_consistency.png")
    P.save(fig_class_balance(df), config.FIGURES_DIR / "03_class_balance.png")
    P.save(fig_features_by_grade(df), config.FIGURES_DIR / "04_features_by_grade.png")
    P.save(fig_missing(df), config.FIGURES_DIR / "05_missing_values.png")
    fig, corr = fig_correlation(df)
    P.save(fig, config.FIGURES_DIR / "06_correlation.png")
    fig, by_group = fig_grade_by_group(df)
    P.save(fig, config.FIGURES_DIR / "07_grade_by_food_group.png")
    P.save(fig_macro_space(df), config.FIGURES_DIR / "08_macro_space.png")

    sample = df.sample(min(len(df), 100_000), random_state=config.SEED)
    filled = sample[NUTRIENTS].fillna(sample[NUTRIENTS].median())
    mi = mutual_info_classif(filled, sample["y"], random_state=config.SEED)

    stats = {
        "rows": len(df),
        "class_share": df["grade"].value_counts(normalize=True).reindex(config.GRADES).round(4).to_dict(),
        "describe": df[NUTRIENTS].describe().round(2).T.to_dict("index"),
        "median_by_grade": df.groupby("grade")[NUTRIENTS].median().round(2).to_dict("index"),
        "spearman_with_grade": corr["y"].drop("y").round(3).to_dict(),
        "mutual_information": dict(zip(NUTRIENTS, np.round(mi, 3).tolist())),
        "grade_share_by_food_group": by_group.round(3).to_dict("index"),
        "food_group_rows": df[config.GROUP_COLUMN].value_counts().to_dict(),
        "created_year_rows": {int(k): int(v) for k, v in df["created_year"].value_counts().sort_index().items()},
        "ukraine_rows": int(df["is_ukraine"].sum()),
        "ukraine_class_share": df.loc[df["is_ukraine"], "grade"].value_counts(normalize=True)
                                 .reindex(config.GRADES).round(4).to_dict(),
        "label_ambiguity": {name: label_ambiguity(df, cols) for name, cols in config.FEATURE_SETS.items()},
    }
    (config.RESULTS_DIR / "eda_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


if __name__ == "__main__":
    result = run()
    print(json.dumps({k: result[k] for k in ("rows", "class_share", "spearman_with_grade",
                                             "mutual_information", "label_ambiguity")},
                     indent=2, ensure_ascii=False))
