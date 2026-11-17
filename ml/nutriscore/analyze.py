"""Step 6. Turn the saved metrics, predictions and models into figures 09–18 and the analysis tables.

    python -m ml.nutriscore.analyze
"""
from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from .. import plotting as P
from . import config, formula
from . import modeling as M
from .eda import FEATURE_LABELS, GRADE_NAMES, GROUP_LABELS

MODEL_LABELS = {
    "Dummy": "Базова (частоти класів)",
    "Formula": "Формула Nutri-Score",
    "LogReg": "Логістична регресія",
    "RandomForest": "Random Forest",
    "LightGBM": "LightGBM",
}
SET_LABELS = {"macros": "4 макронутрієнти", "label": "8 полів етикетки"}
SHORT = {k: v.split(",")[0].lower() for k, v in FEATURE_LABELS.items()}
TRAINED = ["LogReg", "RandomForest", "LightGBM"]
CONFIDENCE_LEVELS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
# Thresholds of the official algorithm, to compare with what the model learned (kcal = kJ / 4.184).
OFFICIAL_THRESHOLDS = {
    "kcal": formula.ENERGY_KJ / formula.KJ_PER_KCAL, "sugars": formula.SUGARS,
    "saturated_fat": formula.SATURATED_FAT, "salt": formula.SALT,
    "proteins": formula.PROTEINS, "fiber": formula.FIBER,
}


def load_predictions(feature_set: str) -> dict[str, np.ndarray]:
    return dict(np.load(config.PROCESSED_DIR / f"predictions_{feature_set}.npz"))


# --------------------------------------------------------------------------- model comparison

def fig_model_comparison(metrics: dict[str, dict], formula_result: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharex=True)
    for ax, (feature_set, result) in zip(axes, metrics.items()):
        rows = [(name, result["models"][name]["test"]["macro_f1"], result["models"][name]["test_ci95"]["macro_f1"])
                for name in M.MODEL_ORDER]
        if feature_set == "label":
            rows.insert(1, ("Formula", formula_result["test"]["macro_f1"], formula_result["test_ci95"]["macro_f1"]))
        rows = rows[::-1]
        values = [r[1] for r in rows]
        ax.barh([MODEL_LABELS[r[0]] for r in rows], values, height=0.5,
                color=[P.MODEL_COLORS[r[0]] for r in rows])
        for y, (_, value, (low, high)) in enumerate(rows):
            ax.plot([low, high], [y, y], color=P.INK, linewidth=1.2, solid_capstyle="butt")
            ax.text(high + 0.015, y, f"{value:.3f}".replace(".", ","), va="center", fontsize=9.5,
                    color=P.INK_SECONDARY)
        ax.set_xlim(0, 1.08)
        ax.set_title(SET_LABELS[feature_set])
        ax.set_xlabel("Macro-F1 на тестовій вибірці (відрізок: 95 % ДІ)")
        P.horizontal_grid(ax)
    fig.tight_layout(w_pad=3)
    return fig


def fig_confusions(result: dict, feature_set: str):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax, name in zip(axes, TRAINED):
        cm = np.array(result["models"][name]["test"]["confusion"], dtype=float)
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=P.SEQUENTIAL, vmin=0, vmax=1)
        for i in range(M.N_CLASSES):
            for j in range(M.N_CLASSES):
                ax.text(j, i, f"{share[i, j] * 100:.0f}", ha="center", va="center", fontsize=9.5,
                        color="#ffffff" if share[i, j] > 0.55 else P.INK)
        ax.set_xticks(range(M.N_CLASSES), GRADE_NAMES)
        ax.set_yticks(range(M.N_CLASSES), GRADE_NAMES)
        ax.set_xlabel("Передбачена оцінка")
        ax.set_title(MODEL_LABELS[name], fontsize=10.5)
        ax.grid(False)
        ax.spines[:].set_visible(False)
    axes[0].set_ylabel("Справжня оцінка")
    fig.suptitle(f"Матриці помилок, % від справжнього класу ({SET_LABELS[feature_set]})",
                 x=0.01, ha="left", fontsize=11.5, fontweight="semibold")
    fig.tight_layout()
    return fig


def fig_per_class(metrics: dict[str, dict]):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
    width = 0.24
    for ax, (feature_set, result) in zip(axes, metrics.items()):
        for k, name in enumerate(TRAINED):
            f1 = [result["models"][name]["test"]["per_class"][g]["f1"] for g in config.GRADES]
            ax.bar(np.arange(M.N_CLASSES) + (k - 1) * (width + 0.02), f1, width=width,
                   color=P.MODEL_COLORS[name], label=MODEL_LABELS[name])
        ax.set_xticks(range(M.N_CLASSES), GRADE_NAMES)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Оцінка Nutri-Score")
        ax.set_title(SET_LABELS[feature_set])
    axes[0].set_ylabel("F1 класу (тест)")
    axes[0].legend(ncol=3, loc="lower left", bbox_to_anchor=(-0.02, 1.12))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- experiments

def variant_label(variant: str) -> str:
    replacements = {
        "label + engineered ratios": "+ похідні співвідношення",
        "label + food group + fruit/veg share": "+ група і частка фруктів/овочів",
        "label + food group": "+ група продуктів",
        "label + fruit/veg share": "+ частка фруктів/овочів",
    }
    if variant in replacements:
        return replacements[variant]
    for prefix, sign in (("macros + ", "+ "), ("label − ", "− ")):
        if variant.startswith(prefix):
            return sign + SHORT[variant[len(prefix):]]
    return SET_LABELS.get(variant, variant)


def fig_ablation(rows: list[dict]):
    base = {r["variant"]: r["macro_f1"] for r in rows if r["kind"] == "base"}
    panels = [
        ("add_one", "macros", "Додаємо одне поле до 4 макронутрієнтів"),
        ("drop_one", "label", "Прибираємо одне поле з 8 полів етикетки"),
        ("beyond_label", "label", "Додаємо те, чого немає на етикетці"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, (kind, reference, title) in zip(axes, panels):
        part = sorted((r for r in rows if r["kind"] == kind), key=lambda r: r["macro_f1"] - base[reference])
        delta = [(r["macro_f1"] - base[reference]) * 100 for r in part]
        ax.barh([variant_label(r["variant"]) for r in part], delta, height=0.5,
                color=[P.BLUE if d >= 0 else P.RED for d in delta])
        span = max(abs(d) for d in delta) or 1
        for y, d in enumerate(delta):
            ax.text(d + (0.03 if d >= 0 else -0.03) * span, y, f"{d:+.1f}".replace("-", "−").replace(".", ","),
                    va="center", ha="left" if d >= 0 else "right", fontsize=9, color=P.INK_SECONDARY)
        # Room for the value labels, so they never run into the row names on the left.
        ax.set_xlim(min(0, min(delta)) - (0.5 if min(delta) < 0 else 0.1) * span,
                    max(0, max(delta)) + (0.5 if max(delta) > 0 else 0.1) * span)
        ax.axvline(0, color=P.AXIS, linewidth=1)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("Зміна macro-F1, в. п. (база {})".format(f"{base[reference]:.3f}".replace(".", ",")))
        P.horizontal_grid(ax)
        ax.spines["left"].set_visible(False)
    fig.tight_layout(w_pad=2)
    return fig


def fig_learning(rows: list[dict]):
    series = [
        ("LightGBM", "label", P.AQUA, "LightGBM, 8 полів"),
        ("LightGBM", "macros", P.BLUE, "LightGBM, 4 макронутрієнти"),
        ("LogReg", "label", P.ORANGE, "Логістична регресія, 8 полів"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for model, feature_set, color, label in series:
        part = [r for r in rows if r["model"] == model and r["feature_set"] == feature_set]
        x = [r["train_rows"] for r in part]
        ax.plot(x, [r["val"]["macro_f1"] for r in part], color=color, marker="o", markersize=5,
                markeredgecolor=P.SURFACE, markeredgewidth=1, label=label)
        ax.plot(x, [r["train"]["macro_f1"] for r in part], color=color, linewidth=1, alpha=0.45)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Розмір навчальної вибірки (лог. шкала)")
    ax.set_ylabel("Macro-F1")
    ax.xaxis.set_major_formatter(lambda v, _: P.thousands(v))
    ax.set_title("Криві навчання: валідація (жирна лінія) та навчальна вибірка (тонка)")
    ax.legend(loc="lower right")
    return fig


def fig_compact(rows: list[dict]):
    """What the model loses when only its first k boosting rounds are kept."""
    colors = {"macros": P.BLUE, "label": P.AQUA}
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for feature_set, color in colors.items():
        part = [r for r in rows if r["feature_set"] == feature_set]
        x = [r["size_mb"] for r in part]
        y = [r["test"]["accuracy"] * 100 for r in part]
        ax.plot(x, y, color=color, marker="o", markersize=5, markeredgecolor=P.SURFACE, markeredgewidth=1,
                label=SET_LABELS[feature_set])
        for r, xi, yi in zip(part, x, y):
            if r["rounds"] in (25, 100, 400) or r["is_full"]:
                ax.annotate(P.thousands(r["rounds"]), (xi, yi), textcoords="offset points", xytext=(0, -14),
                            ha="center", fontsize=8.5, color=P.INK_SECONDARY)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}".replace(".", ","))
    ax.set_xlabel("Розмір моделі, МБ (лог. шкала); підписи точок: кількість ітерацій бустингу")
    ax.set_ylabel("Точність на тестовій вибірці, %")
    ax.set_title("Точність LightGBM залежно від розміру моделі")
    ax.legend(loc="center right")
    return fig


# --------------------------------------------------------------------------- explanations

def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """(rows, features, classes) contributions to the class log-odds."""
    values = shap.TreeExplainer(model).shap_values(X)
    return np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values)


def fig_shap_importance(importances: dict[str, pd.DataFrame]):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1]})
    for ax, (feature_set, table) in zip(axes, importances.items()):
        table = table.loc[table.sum(axis=1).sort_values().index]
        left = np.zeros(len(table))
        for g in config.GRADES:
            ax.barh([FEATURE_LABELS[f].split(",")[0] for f in table.index], table[g], left=left, height=0.55,
                    color=P.GRADE_COLORS[g], edgecolor=P.SURFACE, linewidth=1.2, label=g.upper())
            left += table[g].to_numpy()
        ax.set_title(SET_LABELS[feature_set])
        ax.set_xlabel("Середній |SHAP| (внесок у логарифм шансів класу)")
        P.horizontal_grid(ax)
    axes[0].legend(ncol=5, loc="lower left", bbox_to_anchor=(-0.02, 1.12), handlelength=1, columnspacing=1.3,
                   title=None)
    fig.tight_layout()
    return fig


def fig_shap_dependence(values: np.ndarray, X: pd.DataFrame):
    features = list(X.columns)
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6))
    for ax, feature in zip(axes.ravel(), features):
        j = features.index(feature)
        x = X[feature].to_numpy()
        upper = np.nanpercentile(x, 99)
        for threshold in OFFICIAL_THRESHOLDS.get(feature, []):
            if threshold <= upper:
                ax.axvline(threshold, color=P.GRID, linewidth=0.8, zorder=0)
        for g in ("a", "e"):
            ax.scatter(x, values[:, j, config.GRADES.index(g)], s=3, alpha=0.25, color=P.GRADE_COLORS[g],
                       linewidths=0, rasterized=True)
        ax.axhline(0, color=P.AXIS, linewidth=1)
        ax.set_xlim(0, upper)
        ax.set_title(FEATURE_LABELS[feature], fontsize=10)
        ax.grid(False)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=6, color=P.GRADE_COLORS[g],
                          label=f"внесок у клас {g.upper()}") for g in ("a", "e")]
    fig.legend(handles=handles, ncol=2, loc="upper right", bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Залежність SHAP-внеску від значення ознаки (вертикальні лінії: пороги офіційної формули)",
                 x=0.01, ha="left", fontsize=11.5, fontweight="semibold")
    fig.tight_layout()
    return fig


def explain(df: pd.DataFrame) -> dict:
    importances, out = {}, {}
    for feature_set, features in config.FEATURE_SETS.items():
        model = joblib.load(config.MODELS_DIR / f"{feature_set}_LightGBM.joblib")
        X_test, y_test = M.split_xy(df, "test", features)
        sample = X_test.sample(min(len(X_test), 5_000), random_state=config.SEED)
        values = shap_values(model, sample)
        table = pd.DataFrame(np.abs(values).mean(axis=0), index=features, columns=list(config.GRADES))
        importances[feature_set] = table
        out[feature_set] = {"mean_abs_shap": table.round(4).to_dict("index"),
                            "mean_abs_shap_total": table.sum(axis=1).round(4).to_dict()}
        if feature_set == "label":
            P.save(fig_shap_dependence(values, sample), config.FIGURES_DIR / "15_shap_dependence.png")
            idx = np.random.default_rng(config.SEED).choice(len(X_test), min(len(X_test), 20_000), replace=False)
            perm = {}
            for name in TRAINED:
                fitted = joblib.load(config.MODELS_DIR / f"label_{name}.joblib")
                result = permutation_importance(fitted, X_test.iloc[idx], y_test[idx], scoring="f1_macro",
                                                n_repeats=5, random_state=config.SEED, n_jobs=1)
                perm[name] = dict(zip(features, np.round(result.importances_mean, 4).tolist()))
            out["permutation_importance_label"] = perm
    P.save(fig_shap_importance(importances), config.FIGURES_DIR / "14_shap_importance.png")
    return out


# --------------------------------------------------------------------------- reliability and errors

def reliability(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> dict:
    confidence, correct = proba.max(axis=1), proba.argmax(axis=1) == y
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(confidence, edges) - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            rows.append({"confidence": float(confidence[mask].mean()), "accuracy": float(correct[mask].mean()),
                         "share": float(mask.mean())})
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return {"bins": rows, "ece": round(float(ece), 4)}


def coverage_table(y: np.ndarray, proba: np.ndarray) -> list[dict]:
    confidence, pred = proba.max(axis=1), proba.argmax(axis=1)
    rows = []
    for level in CONFIDENCE_LEVELS:
        keep = confidence >= level
        rows.append({
            "min_confidence": level,
            "coverage": round(float(keep.mean()), 4),
            "accuracy": round(float((pred[keep] == y[keep]).mean()), 4) if keep.any() else None,
            "within_one": round(float((np.abs(pred[keep] - y[keep]) <= 1).mean()), 4) if keep.any() else None,
        })
    return rows


def fig_confidence(predictions: dict[str, dict]):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"macros": P.BLUE, "label": P.AQUA}
    axes[0].plot([0, 1], [0, 1], color=P.AXIS, linewidth=1)
    for feature_set, preds in predictions.items():
        y, proba = preds["y_test"], preds["test_LightGBM"]
        rel = reliability(y, proba)
        axes[0].plot([b["confidence"] for b in rel["bins"]], [b["accuracy"] for b in rel["bins"]],
                     color=colors[feature_set], marker="o", markersize=5, markeredgecolor=P.SURFACE,
                     markeredgewidth=1, label=f"{SET_LABELS[feature_set]} (ECE {rel['ece']:.3f})".replace(".", ","))
        order = np.argsort(-proba.max(axis=1))
        hits = (proba.argmax(axis=1) == y)[order]
        coverage = np.arange(1, len(hits) + 1) / len(hits)
        step = max(1, len(hits) // 400)
        axes[1].plot(coverage[::step] * 100, (np.cumsum(hits) / np.arange(1, len(hits) + 1))[::step] * 100,
                     color=colors[feature_set], label=SET_LABELS[feature_set])
    axes[0].set_xlabel("Упевненість моделі (макс. імовірність)")
    axes[0].set_ylabel("Частка правильних відповідей")
    axes[0].set_title("Калібрування LightGBM")
    axes[0].set_xlim(0.2, 1)
    axes[0].set_ylim(0.2, 1)
    axes[0].legend(loc="upper left")
    axes[1].set_xlabel("Покриття: частка продуктів, яким модель ставить оцінку, %")
    axes[1].set_ylabel("Точність серед оцінених, %")
    axes[1].set_title("Точність проти покриття (спершу найупевненіші)")
    axes[1].set_xlim(0, 100)
    axes[1].legend(loc="lower left")
    fig.tight_layout(w_pad=3)
    return fig


def by_group(test: pd.DataFrame, predictions: dict[str, dict]) -> pd.DataFrame:
    y = test["y"].to_numpy()
    table = pd.DataFrame({"rows": test.groupby(config.GROUP_COLUMN, observed=True).size()})
    hits = {
        "LightGBM · macros": predictions["macros"]["test_LightGBM"].argmax(axis=1) == y,
        "LightGBM · label": predictions["label"]["test_LightGBM"].argmax(axis=1) == y,
        "Formula · label": formula.predict_proba(test).argmax(axis=1) == y,
    }
    for name, hit in hits.items():
        table[name] = pd.Series(hit, index=test.index).groupby(test[config.GROUP_COLUMN], observed=True).mean()
    known = test[config.GROUP_COLUMN].notna().to_numpy()
    table.loc["(no group)"] = [int((~known).sum())] + [float(h[~known].mean()) for h in hits.values()]
    return table


def fig_by_group(table: pd.DataFrame):
    part = table.drop(index="(no group)")
    part = part[part["rows"] >= 30].sort_values("LightGBM · label")  # a handful of rows is noise
    names = [GROUP_LABELS.get(g, g) for g in part.index]
    series = [("Formula · label", P.MODEL_COLORS["Formula"], "Формула, 8 полів"),
              ("LightGBM · macros", P.BLUE, "LightGBM, 4 макронутрієнти"),
              ("LightGBM · label", P.AQUA, "LightGBM, 8 полів")]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    height = 0.24
    for k, (column, color, label) in enumerate(series):
        ax.barh(np.arange(len(part)) + (k - 1) * (height + 0.03), part[column] * 100, height=height,
                color=color, label=label)
    ax.set_yticks(range(len(part)), names)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Точність на тестовій вибірці, %")
    ax.set_title("Точність за групами продуктів", pad=30)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], ncol=3, loc="lower left", bbox_to_anchor=(-0.02, 0.99))
    P.horizontal_grid(ax)
    return fig


def error_analysis(test: pd.DataFrame, predictions: dict[str, dict]) -> dict:
    out = {}
    y = test["y"].to_numpy()
    for feature_set, preds in predictions.items():
        proba = preds["test_LightGBM"]
        pred = proba.argmax(axis=1)
        distance = np.abs(pred - y)
        frame = test.assign(pred=pred, confidence=proba.max(axis=1), distance=distance)
        worst = frame[frame["distance"] >= 3].nlargest(15, "confidence")
        ukraine = test["is_ukraine"].to_numpy()
        out[feature_set] = {
            "error_distance_share": {str(d): round(float((distance == d).mean()), 4) for d in range(5)},
            "direction": {"too_healthy": round(float((pred < y).mean()), 4),
                          "too_unhealthy": round(float((pred > y).mean()), 4)},
            "coverage_table": coverage_table(y, proba),
            "coverage_table_val": coverage_table(preds["y_val"], preds["val_LightGBM"]),
            "reliability": reliability(y, proba),
            "ukraine": ({"rows": int(ukraine.sum()),
                         **{k: round(float(v), 4) for k, v in
                            M.scores_from_confusion(M.confusion(y[ukraine], pred[ukraine])).items()}}
                        if ukraine.sum() >= 50 else {"rows": int(ukraine.sum())}),
            "confident_gross_errors": [
                {"name": str(r.product_name)[:60], "group": None if pd.isna(r.food_group) else str(r.food_group),
                 "true": config.GRADES[r.y].upper(), "predicted": config.GRADES[r.pred].upper(),
                 "confidence": round(float(r.confidence), 3),
                 **{f: (None if pd.isna(getattr(r, f)) else float(getattr(r, f)))
                    for f in config.FEATURE_SETS["label"]}}
                for r in worst.itertuples()
            ],
        }
    return out


def run() -> None:
    config.ensure_dirs()
    P.apply_style()
    df = M.load_dataset()
    test = df[df["split"] == "test"]
    metrics = {fs: M.read_json(f"metrics_{fs}.json") for fs in config.FEATURE_SETS}
    predictions = {fs: load_predictions(fs) for fs in config.FEATURE_SETS}
    formula_result = M.read_json("formula_baseline.json")

    P.save(fig_model_comparison(metrics, formula_result), config.FIGURES_DIR / "09_model_comparison.png")
    for feature_set, result in metrics.items():
        P.save(fig_confusions(result, feature_set), config.FIGURES_DIR / f"10_confusion_{feature_set}.png")
    P.save(fig_per_class(metrics), config.FIGURES_DIR / "11_per_class_f1.png")
    P.save(fig_ablation(M.read_json("ablation.json")), config.FIGURES_DIR / "12_ablation.png")
    P.save(fig_learning(M.read_json("learning_curves.json")), config.FIGURES_DIR / "13_learning_curves.png")
    P.save(fig_compact(M.read_json("compact_models.json")), config.FIGURES_DIR / "18_size_vs_accuracy.png")
    M.write_json("explanations.json", explain(df))
    P.save(fig_confidence(predictions), config.FIGURES_DIR / "16_confidence.png")
    groups = by_group(test, predictions)
    P.save(fig_by_group(groups), config.FIGURES_DIR / "17_accuracy_by_group.png")
    M.write_json("accuracy_by_group.json", groups.round(4).to_dict("index"))
    M.write_json("error_analysis.json", error_analysis(test, predictions))


if __name__ == "__main__":
    run()
