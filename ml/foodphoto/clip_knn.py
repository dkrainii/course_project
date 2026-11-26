"""Step 4d. Recognition by similar photos: CLIP vectors compared with photos of known dishes.

No classifier is trained here. A test photo is compared (cosine similarity of CLIP image vectors)
with a gallery of labelled photos, and the dish of the most similar ones is the answer. This is how
a new dish could be added to the app with a handful of photos and no retraining. Three questions:

* nearest neighbours: the k most similar training photos vote (k chosen on the validation split);
* few photos per dish: each dish is the mean vector ("prototype") of only N random training photos,
  N = 1, 2, 5, 10, 20 (5 random draws each), the setting of a dish added from a few photos;
* photos plus name: the prototype similarity plus the similarity to the dish name prompt of
  clip_embed (zero-shot), which helps most when photos are few.

Uses the train / val / test vectors of clip_embed.py (only the OpenAI models have train.npz).

    python -m ml.foodphoto.clip_knn   → results/clip_knn.json, figures/15_clip_few_photos.png
"""
from __future__ import annotations

import json

import numpy as np

from .. import plotting as P
from . import config
from .analyze import MAIN, SUBSETS, accuracy, load, write
from .clip_analyze import CLIP_LABELS, load_clip

plt = P.plt

MODELS = ["clip_b32", "clip_b16", "clip_l14"]
K_CHOICES = (1, 3, 5, 10, 20, 50)
SHOTS = (1, 2, 5, 10, 20)
DRAWS = 5
CHUNK = 2000


def knn_votes(query: np.ndarray, gallery: np.ndarray, labels: np.ndarray, n_classes: int, ks: tuple) -> dict:
    """Similarity-weighted votes of the k nearest gallery photos, for every k in ks: {k: [query, class]}."""
    out = {k: np.zeros((len(query), n_classes), np.float32) for k in ks}
    kmax = max(ks)
    for i in range(0, len(query), CHUNK):
        sims = query[i:i + CHUNK] @ gallery.T
        idx = np.argpartition(-sims, kmax, axis=1)[:, :kmax]
        part = np.take_along_axis(sims, idx, axis=1)
        order = np.argsort(-part, axis=1)
        idx, part = np.take_along_axis(idx, order, axis=1), np.take_along_axis(part, order, axis=1)
        for k in ks:
            rows = np.repeat(np.arange(len(idx)), k)
            np.add.at(out[k][i:i + CHUNK], (rows, labels[idx[:, :k]].ravel()), part[:, :k].ravel())
    return out


def prototypes(x: np.ndarray, y: np.ndarray, n_classes: int, shots: int | None, rng) -> np.ndarray:
    """Mean (re-normalised) vector of `shots` random photos per class; all photos when shots is None."""
    protos = np.zeros((n_classes, x.shape[1]), np.float32)
    for c in range(n_classes):
        idx = np.flatnonzero(y == c)
        if shots is not None:
            idx = rng.choice(idx, size=min(shots, len(idx)), replace=False)
        protos[c] = x[idx].mean(axis=0)
    return protos / np.linalg.norm(protos, axis=1, keepdims=True)


def scores(pred: np.ndarray, y: np.ndarray, source: np.ndarray) -> dict:
    return {s: accuracy(pred[source == s] == y[source == s]) for s in SUBSETS}


def mean_over_draws(draws: list[dict]) -> dict:
    return {s: {"mean": round(float(np.mean([d[s]["value"] for d in draws])), 4),
                "min": round(float(np.min([d[s]["value"] for d in draws])), 4),
                "max": round(float(np.max([d[s]["value"] for d in draws])), 4)} for s in SUBSETS}


def evaluate(name: str) -> dict:
    c = load_clip(name)
    n = len(c["classes"])
    tr, va, te = c["train"], c["val"], c["test"]
    out = {}

    # nearest neighbours over all training photos, k picked on validation
    val_votes = knn_votes(va["x"], tr["x"], tr["y"], n, K_CHOICES)
    val_acc = {k: float((v.argmax(1) == va["y"]).mean()) for k, v in val_votes.items()}
    k = max(val_acc, key=val_acc.get)
    test_pred = knn_votes(te["x"], tr["x"], tr["y"], n, (k,))[k].argmax(1)
    out["knn"] = {"k": k, "val_top1": {str(a): round(b, 4) for a, b in val_acc.items()},
                  "test": scores(test_pred, te["y"], te["source"])}
    out["prototype_all"] = scores((te["x"] @ prototypes(tr["x"], tr["y"], n, None, None).T).argmax(1),
                                  te["y"], te["source"])
    zero_shot = te["x"] @ c["text"].T
    out["zero_shot"] = scores(zero_shot.argmax(1), te["y"], te["source"])

    # a few photos per dish, with and without the dish name
    out["few"] = {}
    for shots in SHOTS:
        photos, both = [], []
        for draw in range(DRAWS):
            rng = np.random.default_rng(config.SEED + 1000 * shots + draw)
            proto = te["x"] @ prototypes(tr["x"], tr["y"], n, shots, rng).T
            photos.append(scores(proto.argmax(1), te["y"], te["source"]))
            both.append(scores((proto + zero_shot).argmax(1), te["y"], te["source"]))
        out["few"][str(shots)] = {"photos": mean_over_draws(photos), "photos_and_name": mean_over_draws(both)}
        print(f"[knn] {name} {shots:2d} photos: " + "  ".join(
            f"{s} {out['few'][str(shots)]['photos'][s]['mean']:.3f}/{out['few'][str(shots)]['photos_and_name'][s]['mean']:.3f}"
            for s in SUBSETS), flush=True)
    return out


def fig_few(results: dict, cnn: dict, probe: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    model = "clip_l14"
    r = results[model]
    x = list(SHOTS)
    for ax, subset in zip(axes, SUBSETS):
        for key, color, label in [("photos", P.VIOLET, "N фото страви"),
                                  ("photos_and_name", P.AQUA, "N фото + назва страви")]:
            mean = [r["few"][str(s)][key][subset]["mean"] * 100 for s in x]
            low = [r["few"][str(s)][key][subset]["min"] * 100 for s in x]
            high = [r["few"][str(s)][key][subset]["max"] * 100 for s in x]
            ax.plot(x, mean, marker="o", color=color, label=label)
            ax.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
        for value, color, style, label in [
                (r["zero_shot"][subset]["value"], P.ORANGE, "--", "лише назва страви (без фото)"),
                (r["knn"]["test"][subset]["value"], P.INK_SECONDARY, ":", f"найближчі фото з усіх навчальних (k = {r['knn']['k']})"),
                (probe[subset], P.NEUTRAL, "-.", "навчений класифікатор на всіх фото"),
                (cnn[subset], P.BLUE, "-", "EfficientNet-B0 (наша модель, усі фото)")]:
            ax.axhline(value * 100, color=color, linestyle=style, linewidth=1.3, label=label)
        ax.set_xscale("log")
        ax.set_xticks(x, [str(s) for s in x])
        ax.minorticks_off()
        P.horizontal_grid(ax)
        ax.set_title(SUBSETS[subset])
        ax.set_xlabel("Фото на страву, N (середнє з 5 випадкових виборів, смуга — розкид)")
    axes[0].set_ylabel(f"Top-1 на тестовій вибірці, % ({CLIP_LABELS[model]})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    P.save(fig, config.FIGURES_DIR / "15_clip_few_photos.png")


def run() -> dict:
    P.apply_style()
    config.ensure_dirs()
    results = {name: evaluate(name) for name in MODELS}
    write("clip_knn.json", results)
    main = load(MAIN)["test"]
    cnn = {s: float((main["pred"][main["source"] == s] == main["y"][main["source"] == s]).mean()) for s in SUBSETS}
    clip = json.loads((config.RESULTS_DIR / "clip.json").read_text(encoding="utf-8"))
    probe = {s: clip["variants"]["clip_l14/probe"][s]["top1"]["value"] for s in SUBSETS}
    fig_few(results, cnn, probe)
    for name, r in results.items():
        print(f"{name}: knn k={r['knn']['k']} " + " ".join(f"{s} {r['knn']['test'][s]['value']:.3f}" for s in SUBSETS)
              + "  prototype(all) " + " ".join(f"{s} {r['prototype_all'][s]['value']:.3f}" for s in SUBSETS)
              + "  zero-shot " + " ".join(f"{s} {r['zero_shot'][s]['value']:.3f}" for s in SUBSETS))
    return results


if __name__ == "__main__":
    run()
