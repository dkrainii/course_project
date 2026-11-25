"""Step 4. Analysis of the trained runs: every figure and number of the report.

Works from the files the runs leave in RUNS_DIR (predictions.npz, history.json, best.pt), so it
needs no GPU and no Food-101 images; the Ukrainian photos (ml/data/foodphoto/ua/) are used for
the sample and error figures. Runs trained before the Ukrainian dishes were added are read from
RUNS_DIR/food101_only/ for one comparison.

    python -m ml.foodphoto.analyze        → reports/foodphoto/figures/*.png, results/*.json
"""
from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
from matplotlib.ticker import FuncFormatter
from PIL import Image
from scipy.stats import binomtest

from .. import plotting as P
from . import config
from .prepare import folder_rows

plt = P.plt

MAIN = "effb0_finetune"
RUN_ORDER = ["effb0_finetune", "mnv3_finetune", "effb0_nobalance", "effb0_probe"]
RUN_LABELS = {
    "effb0_finetune": "EfficientNet-B0, повне донавчання",
    "mnv3_finetune": "MobileNetV3-Large, повне донавчання",
    "effb0_nobalance": "EfficientNet-B0 без балансування",
    "effb0_probe": "EfficientNet-B0, лише класифікатор",
}
RUN_COLORS = {"effb0_finetune": P.BLUE, "mnv3_finetune": P.ORANGE, "effb0_nobalance": P.VIOLET,
              "effb0_probe": P.NEUTRAL}
DISHES_UK = {"borscht": "Борщ", "deruny": "Деруни", "holubtsi": "Голубці", "olivier_salad": "Олів'є",
             "paska": "Паска", "shashlik": "Шашлик", "varenyky": "Вареники"}
SUBSETS = {"food101": "Food-101", "ua": "Українські страви"}
BOOTSTRAP = 2000


# --------------------------------------------------------------------------- helpers

def num(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def percent_axis(ax, axis: str = "x") -> None:
    fmt = FuncFormatter(lambda v, _: f"{v:.0f}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def load(run: str, base: Path | None = None) -> dict:
    folder = (base or config.RUNS_DIR) / run
    data = np.load(folder / "predictions.npz")
    out = {"classes": json.loads((folder / "classes.json").read_text(encoding="utf-8")),
           "history": json.loads((folder / "history.json").read_text(encoding="utf-8")),
           "metrics": json.loads((folder / "metrics.json").read_text(encoding="utf-8"))}
    for split in ("val", "test"):
        logits = data[f"{split}_logits"].astype(np.float32)
        out[split] = {"logits": logits, "proba": softmax(logits), "y": data[f"{split}_targets"].astype(int),
                      "source": data[f"{split}_sources"], "path": data[f"{split}_paths"]}
        out[split]["pred"] = logits.argmax(axis=1)
        out[split]["rank"] = np.argsort(-logits, axis=1)
    return out


def bootstrap_ci(correct: np.ndarray, seed: int = config.SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = correct[rng.integers(0, len(correct), (BOOTSTRAP, len(correct)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def accuracy(correct: np.ndarray) -> dict:
    low, high = bootstrap_ci(correct)
    return {"n": int(len(correct)), "value": round(float(correct.mean()), 4),
            "ci95": [round(low, 4), round(high, 4)]}


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact McNemar test on paired correctness (same test images)."""
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_a_right": only_a, "only_b_right": only_b, "diff": round(float(a.mean() - b.mean()), 4),
            "p_value": float(f"{p:.3g}")}


def source_names(paths) -> dict[str, dict]:
    with open(config.SOURCES_CSV, encoding="utf-8") as fh:
        rows = {f"ua/{r['dish']}/{Path(r['file']).stem}.jpg": r for r in csv.DictReader(fh)}
    return {p: rows[p] for p in paths if p in rows}


def thumb(path: str, size: int = 220) -> Image.Image:
    with Image.open(config.DATA_DIR / path) as img:
        img = img.convert("RGB")
        side = min(img.size)
        left, top = (img.width - side) // 2, (img.height - side) // 2
        return img.crop((left, top, left + side, top + side)).resize((size, size))


# --------------------------------------------------------------------------- numbers

def dataset_counts(main: dict) -> dict:
    ua = folder_rows(config.UA_DIR, "ua")
    counts = {d: {s: 0 for s in ("train", "val", "test")} for d in DISHES_UK}
    for row in ua:
        counts[row["label"]][row["split"]] += 1
    n_test_ua = int((main["test"]["source"] == "ua").sum())
    assert n_test_ua == sum(c["test"] for c in counts.values()), "local photos differ from the Colab run"
    n_food101 = len(main["classes"]) - len(DISHES_UK)
    return {
        "ua": counts,
        "ua_total": {s: sum(c[s] for c in counts.values()) for s in ("train", "val", "test")},
        "food101_classes": n_food101,
        "food101_per_class": {"train": 750 - config.VAL_PER_FOOD101_CLASS, "val": config.VAL_PER_FOOD101_CLASS,
                              "test": 250},
        "food101_total": {"train": n_food101 * (750 - config.VAL_PER_FOOD101_CLASS),
                          "val": int((main["val"]["source"] == "food101").sum()),
                          "test": int((main["test"]["source"] == "food101").sum())},
    }


def run_metrics(runs: dict[str, dict]) -> dict:
    out = {}
    for name, r in runs.items():
        t = r["test"]
        entry = {"best_epoch": r["metrics"]["best_epoch"],
                 "train_minutes": round(sum(e["seconds"] for e in r["history"]["history"]) / 60, 1),
                 "val_top1": r["metrics"]["val"]["top1"]}
        for subset in ("all", *SUBSETS):
            mask = np.ones(len(t["y"]), bool) if subset == "all" else t["source"] == subset
            entry[subset] = {"top1": accuracy(t["pred"][mask] == t["y"][mask]),
                             "top5": accuracy((t["rank"][mask, :5] == t["y"][mask, None]).any(axis=1))}
        out[name] = entry
    return out


def comparisons(runs: dict[str, dict], food101_only: dict[str, dict]) -> list[dict]:
    pairs = [("effb0_finetune", "effb0_probe", "повне донавчання проти лише класифікатора"),
             ("effb0_finetune", "mnv3_finetune", "EfficientNet-B0 проти MobileNetV3-Large"),
             ("effb0_finetune", "effb0_nobalance", "балансування класів увімкнене проти вимкненого")]
    out = []
    for a, b, title in pairs:
        ta, tb = runs[a]["test"], runs[b]["test"]
        for subset in SUBSETS:
            m = ta["source"] == subset
            out.append({"a": a, "b": b, "question": title, "subset": subset,
                        **mcnemar(ta["pred"][m] == ta["y"][m], tb["pred"][m] == tb["y"][m])})
    for name, base in food101_only.items():  # do 7 extra classes cost Food-101 accuracy?
        ta, tb = runs[name]["test"], base["test"]
        m = ta["source"] == "food101"
        assert (ta["path"][m] == tb["path"]).all()
        out.append({"a": name, "b": f"food101_only/{name}", "question": "108 класів проти 101 (лише Food-101)",
                    "subset": "food101", **mcnemar(ta["pred"][m] == ta["y"][m], tb["pred"] == tb["y"])})
    return out


def per_dish(r: dict) -> list[dict]:
    t, classes = r["test"], r["classes"]
    rows = []
    for dish in DISHES_UK:
        i = classes.index(dish)
        true, said = t["y"] == i, t["pred"] == i
        wrong = t["pred"][true & ~said]
        fp_sources = t["y"][said & ~true]
        rows.append({
            "dish": dish, "n_test": int(true.sum()),
            "recall": round(float(said[true].mean()), 4),
            "precision": round(float(true[said].mean()), 4) if said.any() else None,
            "top5": round(float((t["rank"][true, :5] == i).any(axis=1).mean()), 4),
            "false_positives": int((said & ~true).sum()),
            "taken_for": sorted(({classes[j]: int((wrong == j).sum()) for j in set(wrong)}).items(),
                                key=lambda kv: -kv[1]),
            "false_positive_from": sorted(({classes[j]: int((fp_sources == j).sum()) for j in set(fp_sources)}).items(),
                                          key=lambda kv: -kv[1])[:5],
        })
    return rows


def food101_classes(r: dict) -> list[dict]:
    t, classes = r["test"], r["classes"]
    rows = []
    for i, name in enumerate(classes[:-len(DISHES_UK)]):
        m = t["y"] == i
        wrong = t["pred"][m & (t["pred"] != i)]
        top = np.bincount(wrong, minlength=len(classes)).argmax() if len(wrong) else None
        rows.append({"class": name, "top1": round(float((t["pred"][m] == i).mean()), 4),
                     "most_confused_with": classes[top] if top is not None else None,
                     "confused_count": int((wrong == top).sum()) if top is not None else 0})
    return sorted(rows, key=lambda row: row["top1"])


def topk_curve(r: dict, k_max: int = 10) -> dict:
    t = r["test"]
    out = {}
    for subset in SUBSETS:
        m = t["source"] == subset
        hit = t["rank"][m, :k_max] == t["y"][m, None]
        out[subset] = [round(float(hit[:, :k].any(axis=1).mean()), 4) for k in range(1, k_max + 1)]
    return out


def calibration(r: dict, bins: int = 15) -> dict:
    t = r["test"]
    conf, correct = t["proba"].max(axis=1), t["pred"] == t["y"]
    edges = np.linspace(0, 1, bins + 1)
    table, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
            table.append({"from": round(float(lo), 3), "to": round(float(hi), 3), "n": int(m.sum()),
                          "confidence": round(float(conf[m].mean()), 4), "accuracy": round(float(correct[m].mean()), 4)})
    thresholds = []
    for threshold in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        out = {"threshold": threshold}
        for subset in SUBSETS:
            m = t["source"] == subset
            keep = conf[m] >= threshold
            out[subset] = {"coverage": round(float(keep.mean()), 4),
                           "accuracy": round(float(correct[m][keep].mean()), 4) if keep.any() else None,
                           "top3_rest": round(float((t["rank"][m][~keep, :3] == t["y"][m][~keep, None])
                                                    .any(axis=1).mean()), 4) if (~keep).any() else None}
        thresholds.append(out)
    return {"ece": round(float(ece), 4), "mean_confidence": round(float(conf.mean()), 4),
            "accuracy": round(float(correct.mean()), 4), "bins": table, "thresholds": thresholds}


def ua_errors(r: dict) -> list[dict]:
    t, classes = r["test"], r["classes"]
    m = (t["source"] == "ua") & (t["pred"] != t["y"])
    rows = []
    for i in np.flatnonzero(m):
        rows.append({"path": str(t["path"][i]), "true": classes[t["y"][i]], "pred": classes[t["pred"][i]],
                     "confidence": round(float(t["proba"][i].max()), 3),
                     "true_rank": int(np.flatnonzero(t["rank"][i] == t["y"][i])[0]) + 1})
    return sorted(rows, key=lambda row: -row["confidence"])


def speed() -> dict:
    """Size and single-image CPU latency of the two architectures (this machine)."""
    import torch

    from .train import build_model

    torch.set_num_threads(4)
    out = {"cpu": platform.processor() or platform.machine(), "threads": 4, "torch": torch.__version__}
    for name in ("effb0_finetune", "mnv3_finetune"):
        ckpt = torch.load(config.RUNS_DIR / name / "best.pt", map_location="cpu", weights_only=False)
        model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        x = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        times = []
        with torch.inference_mode():
            for i in range(220):  # 20 warm-up runs, then 200 timed ones
                started = time.perf_counter()
                model(x)
                if i >= 20:
                    times.append(time.perf_counter() - started)
        params = sum(p.numel() for p in model.parameters())
        out[name] = {"arch": ckpt["arch"], "params_m": round(params / 1e6, 2),
                     "size_mb_fp32": round(params * 4 / 2**20, 1),
                     "latency_ms_median": round(float(np.median(times)) * 1000, 1),
                     "latency_ms_p90": round(float(np.percentile(times, 90)) * 1000, 1)}
    return out


# --------------------------------------------------------------------------- figures

def fig_dataset(counts: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    dishes = sorted(DISHES_UK, key=lambda d: sum(counts["ua"][d].values()))
    left = np.zeros(len(dishes))
    for split, color, label in (("train", P.BLUE, "навчальна"), ("val", P.AQUA, "валідаційна"),
                                ("test", P.ORANGE, "тестова")):
        values = np.array([counts["ua"][d][split] for d in dishes])
        ax.barh([DISHES_UK[d] for d in dishes], values, left=left, color=color, label=label, height=0.62)
        left += values
    for i, total in enumerate(left):
        ax.text(total + 4, i, f"{total:.0f}", va="center", fontsize=9, color=P.INK_SECONDARY)
    P.horizontal_grid(ax)
    ax.set_xlabel("Кількість фотографій")
    ax.set_title(f"Українські страви: {int(left.sum())} фото з Wikimedia Commons після ручного відбору", pad=24)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    ax.set_xlim(0, left.max() * 1.12)
    P.save(fig, config.FIGURES_DIR / "01_dataset.png")


def fig_samples(per_dish_count: int = 5) -> list[dict]:
    rows = [r for r in folder_rows(config.UA_DIR, "ua") if r["split"] == "train"]
    chosen = []
    for dish in DISHES_UK:
        items = [r for r in rows if r["label"] == dish]
        step = max(1, len(items) // per_dish_count)
        chosen += [(dish, r["path"]) for r in items[::step][:per_dish_count]]
    fig, axes = plt.subplots(len(DISHES_UK), per_dish_count, figsize=(per_dish_count * 1.5, len(DISHES_UK) * 1.55))
    for ax, (dish, path) in zip(axes.flat, chosen):
        ax.imshow(thumb(path))
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for row, dish in enumerate(DISHES_UK):
        axes[row, 0].set_ylabel(DISHES_UK[dish], rotation=0, ha="right", va="center", fontsize=10, color=P.INK)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    P.save(fig, config.FIGURES_DIR / "02_samples.png")
    names = source_names([p for _, p in chosen])
    return [{"figure": "02_samples.png", "dish": d, **{k: names[p][k] for k in ("title", "author", "licence", "url")}}
            for d, p in chosen]


def fig_learning(runs: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for name in RUN_ORDER:
        h = runs[name]["history"]["history"]
        axes[0].plot([e["epoch"] for e in h], [e["val_top1"] * 100 for e in h], color=RUN_COLORS[name],
                     label=RUN_LABELS[name], marker="o", markersize=3.5)
    axes[0].set_xlabel("Епоха")
    axes[0].set_ylabel("Top-1 на валідації, %")
    axes[0].set_title("Точність на валідаційній вибірці")
    axes[0].legend(loc="lower right", fontsize=8.5)
    h = runs[MAIN]["history"]["history"]
    epochs = [e["epoch"] for e in h]
    axes[1].plot(epochs, [e["train_loss"] for e in h], color=P.BLUE, label="навчальна (з аугментацією)")
    axes[1].plot(epochs, [e["val_loss"] for e in h], color=P.ORANGE, label="валідаційна")
    ax2 = axes[1].twinx()
    ax2.plot(epochs, [e["lr"] for e in h], color=P.NEUTRAL, linestyle="--",
             linewidth=1.3, label="швидкість навчання")
    ax2.set_ylabel("Швидкість навчання класифікатора, ×10⁻³", color=P.MUTED)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: num(v * 1000, 1)))
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    axes[1].set_xlabel("Епоха")
    axes[1].set_ylabel("Функція втрат (крос-ентропія)")
    axes[1].set_title(RUN_LABELS[MAIN] + ": функція втрат")
    lines = axes[1].get_legend_handles_labels()
    extra = ax2.get_legend_handles_labels()
    axes[1].legend(lines[0] + extra[0], lines[1] + extra[1], loc="upper right", fontsize=8.5)
    fig.tight_layout()
    P.save(fig, config.FIGURES_DIR / "03_learning_curves.png")


def fig_comparison(metrics: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    names = RUN_ORDER[::-1]
    for ax, subset in zip(axes, SUBSETS):
        for i, name in enumerate(names):
            m = metrics[name][subset]["top1"]
            value, (low, high) = m["value"] * 100, (v * 100 for v in m["ci95"])
            ax.barh(i, value, color=RUN_COLORS[name], height=0.6)
            ax.errorbar(value, i, xerr=[[value - low], [high - value]], color=P.INK, capsize=3, linewidth=1)
            ax.text(high + 0.8, i, num(value), va="center", fontsize=9)
        P.horizontal_grid(ax)
        ax.set_xlim(50, 100)
        n = metrics[MAIN][subset]["top1"]["n"]
        ax.set_title(f"{SUBSETS[subset]} (n = {P.thousands(n)})")
        ax.set_xlabel("Top-1 на тестовій вибірці, % (відрізок: 95 % ДІ)")
    axes[0].set_yticks(range(len(names)), [RUN_LABELS[n] for n in names])
    fig.tight_layout()
    P.save(fig, config.FIGURES_DIR / "04_model_comparison.png")


def fig_per_dish(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: r["recall"])
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    y = np.arange(len(rows))
    ax.barh(y + 0.19, [r["recall"] * 100 for r in rows], height=0.36, color=P.BLUE, label="повнота (recall)")
    ax.barh(y - 0.19, [r["precision"] * 100 for r in rows], height=0.36, color=P.AQUA, label="влучність (precision)")
    for i, r in enumerate(rows):
        ax.text(r["recall"] * 100 + 0.8, i + 0.19, num(r["recall"] * 100, 0), va="center", fontsize=8.5)
        ax.text(r["precision"] * 100 + 0.8, i - 0.19, num(r["precision"] * 100, 0), va="center", fontsize=8.5)
    ax.set_yticks(y, [f"{DISHES_UK[r['dish']]} (n = {r['n_test']})" for r in rows])
    P.horizontal_grid(ax)
    ax.set_xlim(0, 108)
    ax.set_xlabel("%, тестова вибірка")
    ax.set_title("Українські страви: повнота і влучність основної моделі", pad=24)
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    P.save(fig, config.FIGURES_DIR / "05_ua_per_dish.png")


def fig_ua_confusion(r: dict) -> dict:
    t, classes = r["test"], r["classes"]
    dishes = list(DISHES_UK)
    index = [classes.index(d) for d in dishes]
    matrix = np.zeros((len(dishes) + 1, len(dishes) + 1), int)  # last row/column: any Food-101 class
    to_col = {c: k for k, c in enumerate(index)}
    for y, p, src in zip(t["y"], t["pred"], t["source"]):
        row = to_col.get(y, len(dishes))
        col = to_col.get(p, len(dishes))
        if row == len(dishes) and col == len(dishes):
            continue  # Food-101 → Food-101 is figure 07
        matrix[row, col] += 1
    labels = [DISHES_UK[d] for d in dishes] + ["Food-101 (будь-який)"]
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    shown = matrix[:-1] / matrix[:-1].sum(axis=1, keepdims=True)
    ax.imshow(np.vstack([shown, np.zeros((1, len(labels)))]), cmap=P.SEQUENTIAL, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == len(dishes) and j == len(dishes):
                continue
            value = matrix[i, j]
            if value:
                dark = i < len(dishes) and shown[i, j] > 0.5
                ax.text(j, i, str(value), ha="center", va="center", fontsize=9,
                        color="white" if dark else P.INK)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Передбачений клас")
    ax.set_ylabel("Справжній клас")
    ax.grid(False)
    ax.set_title("Матриця помилок для українських страв (кількість фото)")
    P.save(fig, config.FIGURES_DIR / "06_ua_confusion.png")
    return {"labels": dishes + ["food101"], "matrix": matrix.tolist()}


def fig_food101_classes(rows: list[dict], show: int = 12) -> None:
    worst, best = rows[:show], rows[-show:]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    panels = ((axes[0], worst[::-1], f"{show} найскладніших класів", P.ORANGE, True),
              (axes[1], best, f"{show} найлегших класів", P.BLUE, False))
    for ax, part, title, color, with_confusion in panels:
        values = [r["top1"] * 100 for r in part]
        ax.barh([r["class"].replace("_", " ") for r in part], values, color=color, height=0.62)
        for i, (v, r) in enumerate(zip(values, part)):
            note = num(v, 0)
            if with_confusion and r["most_confused_with"]:
                note += f"  → {r['most_confused_with'].replace('_', ' ')}"
            ax.text(v + 1, i, note, va="center", fontsize=8.3, color=P.INK_SECONDARY)
        P.horizontal_grid(ax)
        ax.set_xlim(0, 140 if with_confusion else 110)
        ax.set_title(title)
        ax.set_xlabel("Top-1, % (→ клас, з яким плутається найчастіше)" if with_confusion else "Top-1, %")
    fig.tight_layout()
    P.save(fig, config.FIGURES_DIR / "08_food101_classes.png")


def fig_topk(curves: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    k = np.arange(1, 11)
    for name, style in (("effb0_finetune", "-"), ("mnv3_finetune", "--")):
        for subset, color in (("food101", P.BLUE), ("ua", P.ORANGE)):
            ax.plot(k, np.array(curves[name][subset]) * 100, style, color=color, marker="o", markersize=3,
                    label=f"{SUBSETS[subset]}, {RUN_LABELS[name].split(',')[0]}")
    ax.set_xticks(k)
    ax.set_xlabel("k (скільки варіантів показано користувачу)")
    ax.set_ylabel("Top-k на тестовій вибірці, %")
    ax.set_title("Частка фото, для яких правильна страва серед перших k варіантів")
    ax.legend(fontsize=8.5, loc="lower right")
    P.save(fig, config.FIGURES_DIR / "09_topk.png")


def fig_confidence(cal: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    bins = cal["bins"]
    axes[0].plot([0, 1], [0, 1], color=P.NEUTRAL, linestyle="--", linewidth=1.2, label="ідеальне калібрування")
    axes[0].plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], color=P.BLUE, marker="o",
                 markersize=4, label="основна модель")
    axes[0].set_xlabel("Упевненість моделі (максимальна ймовірність)")
    axes[0].set_ylabel("Частка правильних відповідей")
    axes[0].set_title(f"Калібрування (ECE = {num(cal['ece'], 3)})")
    axes[0].legend(loc="upper left", fontsize=9)
    for subset, color in (("food101", P.BLUE), ("ua", P.ORANGE)):
        pts = [t for t in cal["thresholds"] if t[subset]["accuracy"] is not None]
        axes[1].plot([t[subset]["coverage"] * 100 for t in pts], [t[subset]["accuracy"] * 100 for t in pts],
                     color=color, marker="o", markersize=4, label=SUBSETS[subset])
        if subset == "food101":
            for t in pts:
                if t["threshold"] in (0.5, 0.8):
                    axes[1].annotate(f"поріг {num(t['threshold'])}",
                                     (t[subset]["coverage"] * 100, t[subset]["accuracy"] * 100),
                                     textcoords="offset points", xytext=(6, 6), fontsize=8, color=P.MUTED)
    comma = FuncFormatter(lambda v, _: num(v, 1))
    for axis in (axes[0].xaxis, axes[0].yaxis):
        axis.set_major_formatter(comma)
    axes[1].set_xlabel("Покриття: частка фото, де модель упевнена ≥ порогу, %")
    axes[1].set_ylabel("Top-1 серед них, %")
    axes[1].set_title("Точність проти покриття")
    axes[1].legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    P.save(fig, config.FIGURES_DIR / "10_confidence.png")


def fig_errors(errors: list[dict], columns: int = 6) -> list[dict]:
    shown = errors[:columns * 3]
    rows_n = (len(shown) - 1) // columns + 1
    fig, axes = plt.subplots(rows_n, columns, figsize=(columns * 1.9, rows_n * 2.35))
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, e in zip(axes.flat, shown):
        ax.imshow(thumb(e["path"]))
        pred = DISHES_UK.get(e["pred"], e["pred"].replace("_", " "))
        ax.set_title(f"{DISHES_UK[e['true']]}\n→ {pred} ({num(e['confidence'] * 100, 0)} %)", fontsize=8.2,
                     loc="center", pad=3, fontweight="normal")
    fig.suptitle("Помилки на українських стравах: справжня страва → відповідь моделі (упевненість)",
                 x=0.02, ha="left", fontsize=10.5, fontweight="semibold")
    fig.tight_layout()
    P.save(fig, config.FIGURES_DIR / "07_ua_errors.png")
    names = source_names([e["path"] for e in shown])
    return [{"figure": "07_ua_errors.png", "dish": e["true"],
             **{k: names[e["path"]][k] for k in ("title", "author", "licence", "url")}} for e in shown]


def fig_speed(sp: dict, metrics: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for name in ("effb0_finetune", "mnv3_finetune"):
        x, y = sp[name]["latency_ms_median"], metrics[name]["all"]["top1"]["value"] * 100
        ax.scatter(x, y, s=sp[name]["size_mb_fp32"] * 12, color=RUN_COLORS[name], alpha=0.85)
        ax.annotate(f"{RUN_LABELS[name].split(',')[0]}\n{num(sp[name]['params_m'], 1)} млн параметрів, "
                    f"{num(sp[name]['size_mb_fp32'], 1)} МБ", (x, y), textcoords="offset points", xytext=(14, -4),
                    fontsize=8.5, va="center")
    ax.set_xlabel(f"Час розпізнавання одного фото на CPU, мс (медіана, {sp['threads']} потоки)")
    ax.set_ylabel("Top-1 на всій тестовій вибірці, %")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: num(v, 1)))
    ax.set_title("Точність проти швидкості (розмір кола ∝ розмір моделі)")
    ax.set_xlim(0, max(sp[n]["latency_ms_median"] for n in ("effb0_finetune", "mnv3_finetune")) * 2)
    ys = [metrics[n]["all"]["top1"]["value"] * 100 for n in ("effb0_finetune", "mnv3_finetune")]
    ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)
    P.save(fig, config.FIGURES_DIR / "11_speed.png")


# --------------------------------------------------------------------------- run

def write(name: str, data) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / name).write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def run() -> None:
    P.apply_style()
    config.ensure_dirs()
    runs = {name: load(name) for name in RUN_ORDER}
    base = config.RUNS_DIR / "food101_only"
    food101_only = {n: load(n, base) for n in ("effb0_finetune", "mnv3_finetune") if (base / n / "metrics.json").exists()}
    main = runs[MAIN]

    counts = dataset_counts(main)
    metrics = run_metrics(runs)
    for name, r in food101_only.items():
        t = r["test"]
        metrics[f"food101_only/{name}"] = {"food101": {"top1": accuracy(t["pred"] == t["y"]),
                                                       "top5": accuracy((t["rank"][:, :5] == t["y"][:, None]).any(1))}}
    tests = comparisons(runs, food101_only)
    dishes = per_dish(main)
    classes101 = food101_classes(main)
    curves = {name: topk_curve(runs[name]) for name in ("effb0_finetune", "mnv3_finetune")}
    cal = calibration(main)
    errors = ua_errors(main)
    sp = speed()

    write("dataset.json", counts)
    write("metrics.json", metrics)
    write("significance.json", tests)
    write("ua_per_dish.json", {name: per_dish(runs[name]) for name in RUN_ORDER})
    write("food101_classes.json", classes101)
    write("topk.json", curves)
    write("calibration.json", cal)
    write("ua_errors.json", errors)
    write("speed.json", sp)

    fig_dataset(counts)
    credits = fig_samples()
    fig_learning(runs)
    fig_comparison(metrics)
    fig_per_dish(dishes)
    write("ua_confusion.json", fig_ua_confusion(main))
    fig_food101_classes(classes101)
    fig_topk(curves)
    fig_confidence(cal)
    credits += fig_errors(errors)
    fig_speed(sp, metrics)
    with open(config.RESULTS_DIR / "figure_photo_sources.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["figure", "dish", "title", "author", "licence", "url"])
        writer.writeheader()
        writer.writerows(credits)

    for name in RUN_ORDER:
        m = metrics[name]
        print(f"[analyze] {name:16} Food-101 {m['food101']['top1']['value']:.4f}  UA {m['ua']['top1']['value']:.4f}")
    for t in tests:
        print(f"[analyze] {t['a']} vs {t['b']} ({t['subset']}): diff {t['diff']:+.4f}, p = {t['p_value']}")
    print(f"[analyze] figures in {config.FIGURES_DIR}")


if __name__ == "__main__":
    run()
