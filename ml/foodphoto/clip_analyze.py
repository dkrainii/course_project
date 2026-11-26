"""Step 4b. CLIP against the trained CNN, on the same test photos.

Reads the vectors of clip_embed.py (RUNS_DIR/clip_*/) and the CNN predictions of train.py
(RUNS_DIR/effb0_finetune/predictions.npz) and compares, for each CLIP model:

* zero-shot     the class whose name ("a photo of {dish}, a type of food.") is closest to the photo;
                CLIP never sees our training photos;
* linear probe  a softmax classifier on the CLIP vectors of our training photos (weight decay
                chosen on the validation split, the same class balancing as the CNN);
* hybrid        the CNN answers when its top probability is at least the app threshold (0.5),
                CLIP (L/14) otherwise;
* average       the mean of the CNN and CLIP (L/14) probabilities.

Each variant gets top-1 / top-3 on Food-101 and on the Ukrainian dishes with a bootstrap CI and an
exact McNemar test against the CNN, plus model size and single-photo CPU latency.

    python -m ml.foodphoto.clip_analyze   → results/clip.json, figures/12_clip_comparison.png, 13_clip_speed.png
"""
from __future__ import annotations

import json
import platform
import time

import numpy as np
import torch
from matplotlib.ticker import FuncFormatter

from .. import plotting as P
from . import config
from .analyze import MAIN, SUBSETS, accuracy, load, mcnemar, num, softmax, write
from .train import BALANCE_POWER

plt = P.plt

CLIP_MODELS = ["clip_b32", "clip_b16", "clip_l14"]
CLIP_LABELS = {"clip_b32": "CLIP ViT-B/32", "clip_b16": "CLIP ViT-B/16", "clip_l14": "CLIP ViT-L/14"}
BEST_CLIP = "clip_l14"
LOGIT_SCALE = 100.0           # CLIP's learned temperature for image–text similarities
CONFIDENT = 0.5               # the app's "most likely" threshold (meta.json of export.py)
WEIGHT_DECAYS = (0.0, 1e-7, 1e-6, 1e-5, 1e-4)


# --------------------------------------------------------------------------- data

def load_clip(name: str) -> dict:
    folder = config.RUNS_DIR / name
    text = np.load(folder / "text.npz")
    out = {"text": text["embeddings"].astype(np.float32), "classes": [str(c) for c in text["classes"]],
           "prompts": [str(p) for p in text["prompts"]]}
    for split in ("train", "val", "test"):
        d = np.load(folder / f"{split}.npz")
        out[split] = {"x": d["embeddings"].astype(np.float32), "y": d["targets"].astype(int),
                      "source": d["sources"], "path": d["paths"]}
    return out


# --------------------------------------------------------------------------- linear probe

def fit_probe(x: np.ndarray, y: np.ndarray, n_classes: int, weight_decay: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Multinomial logistic regression, full-batch L-BFGS; class weights 1/n^BALANCE_POWER as in train.py."""
    torch.manual_seed(config.SEED)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    counts = torch.bincount(yt, minlength=n_classes).double()
    weights = (1.0 / counts[yt] ** BALANCE_POWER).float()
    weights /= weights.mean()
    w = torch.zeros(x.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=500, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.cross_entropy(xt @ w + b, yt, reduction="none") * weights).mean()
        loss = loss + weight_decay * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def probe_logits(c: dict) -> tuple[np.ndarray, dict]:
    n = len(c["classes"])
    scores, best = {}, None
    for wd in WEIGHT_DECAYS:
        w, b = fit_probe(c["train"]["x"], c["train"]["y"], n, wd)
        val = (torch.from_numpy(c["val"]["x"]) @ w + b).argmax(1).numpy()
        scores[wd] = float((val == c["val"]["y"]).mean())
        if best is None or scores[wd] > scores[best[0]]:
            best = (wd, w, b)
        print(f"[clip] probe wd={wd:g}: val top-1 {scores[wd]:.4f}", flush=True)
    wd, w, b = best
    logits = (torch.from_numpy(c["test"]["x"]) @ w + b).numpy()
    return logits, {"weight_decay": wd, "val_top1": {f"{k:g}": round(v, 4) for k, v in scores.items()}}


# --------------------------------------------------------------------------- metrics

def scores(proba: np.ndarray, y: np.ndarray, source: np.ndarray) -> dict:
    rank = np.argsort(-proba, axis=1)[:, :3]
    out = {}
    for subset in ("all", *SUBSETS):
        m = np.ones(len(y), bool) if subset == "all" else source == subset
        out[subset] = {"top1": accuracy(rank[m, 0] == y[m]), "top3": accuracy((rank[m] == y[m, None]).any(1))}
    return out


def speed(models: list[str]) -> dict:
    """Single-photo CPU latency of each image encoder and of the CNN, measured in one session (4 threads)."""
    import open_clip

    from .clip_embed import MODELS
    from .train import build_model

    torch.set_num_threads(4)
    out = {"cpu": platform.processor() or platform.machine(), "threads": 4, "torch": torch.__version__}

    def timed(fn, x) -> dict:
        times = []
        with torch.inference_mode():
            for i in range(70):  # 10 warm-up runs, then 60 timed ones
                started = time.perf_counter()
                fn(x)
                if i >= 10:
                    times.append(time.perf_counter() - started)
        return {"latency_ms_median": round(float(np.median(times)) * 1000, 1),
                "latency_ms_p90": round(float(np.percentile(times, 90)) * 1000, 1)}

    ckpt = torch.load(config.RUNS_DIR / MAIN / "best.pt", map_location="cpu", weights_only=False)
    cnn = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    cnn.load_state_dict(ckpt["model"])
    params = sum(p.numel() for p in cnn.eval().parameters())
    out[MAIN] = {"params_m": round(params / 1e6, 1), "size_mb_fp32": round(params * 4 / 2**20, 1),
                 **timed(cnn, torch.randn(1, 3, 224, 224))}
    del cnn
    for name in models:
        arch, weights = MODELS[name]
        model = open_clip.create_model(arch, pretrained=weights).eval()
        visual = model.visual  # only the image encoder runs per photo; class-name vectors are precomputed
        del model
        params = sum(p.numel() for p in visual.parameters())
        size = visual.image_size if isinstance(visual.image_size, int) else visual.image_size[0]
        out[name] = {"params_m": round(params / 1e6, 1), "size_mb_fp32": round(params * 4 / 2**20, 1),
                     **timed(visual, torch.randn(1, 3, size, size))}
        print(f"[clip] speed {name}: {out[name]}", flush=True)
        del visual
    return out


# --------------------------------------------------------------------------- figures

VARIANT_ORDER = [MAIN, "clip_b32/zero_shot", "clip_b16/zero_shot", "clip_l14/zero_shot",
                 "clip_b32/probe", "clip_b16/probe", "clip_l14/probe", "hybrid", "average"]


def variant_label(key: str) -> str:
    if key == MAIN:
        return "EfficientNet-B0 (наша модель)"
    if key == "hybrid":
        return "Наша модель, за невпевненості CLIP L/14"
    if key == "average":
        return "Середнє нашої моделі й CLIP L/14"
    model, kind = key.split("/")
    return f"{CLIP_LABELS[model]}, {'без навчання' if kind == 'zero_shot' else 'навчений класифікатор'}"


def variant_color(key: str) -> str:
    if key == MAIN:
        return P.BLUE
    if key in ("hybrid", "average"):
        return P.AQUA
    return P.ORANGE if key.endswith("zero_shot") else P.VIOLET


def fig_comparison(variants: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    names = VARIANT_ORDER[::-1]
    for ax, subset in zip(axes, SUBSETS):
        for i, key in enumerate(names):
            m = variants[key][subset]["top1"]
            value, (low, high) = m["value"] * 100, (v * 100 for v in m["ci95"])
            ax.barh(i, value, color=variant_color(key), height=0.6)
            ax.errorbar(value, i, xerr=[[value - low], [high - value]], color=P.INK, capsize=3, linewidth=1)
            ax.text(high + 0.8, i, num(value), va="center", fontsize=9)
        main = variants[MAIN][subset]["top1"]["value"] * 100
        ax.axvline(main, color=P.BLUE, linewidth=1, linestyle=":")
        P.horizontal_grid(ax)
        ax.set_xlim(50, 100)
        ax.set_title(f"{SUBSETS[subset]} (n = {P.thousands(variants[MAIN][subset]['top1']['n'])})")
        ax.set_xlabel("Top-1 на тестовій вибірці, % (відрізок: 95 % ДІ)")
    axes[0].set_yticks(range(len(names)), [variant_label(n) for n in names])
    fig.tight_layout()
    P.save(fig, config.FIGURES_DIR / "12_clip_comparison.png")


def fig_speed(sp: dict, variants: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    points = [(MAIN, MAIN, "EfficientNet-B0")] + [(m, f"{m}/zero_shot", CLIP_LABELS[m]) for m in CLIP_MODELS]
    for model, key, label in points:
        x, y = sp[model]["latency_ms_median"], variants[key]["all"]["top1"]["value"] * 100
        ax.scatter(x, y, s=max(sp[model]["size_mb_fp32"], 15) * 1.2, color=variant_color(key), alpha=0.85)
        ax.annotate(f"{label}\n{num(sp[model]['params_m'], 1)} млн параметрів, {num(sp[model]['size_mb_fp32'], 0)} МБ",
                    (x, y), textcoords="offset points", xytext=(12, 0), fontsize=8.5, va="center")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(f"Час обробки одного фото на CPU, мс (медіана, {sp['threads']} потоки, PyTorch; шкала логарифмічна)")
    ax.set_ylabel("Top-1 на всій тестовій вибірці, %")
    ax.set_title("Точність проти швидкості: наша модель і CLIP без навчання (розмір кола ∝ розмір моделі)")
    xs = [sp[m]["latency_ms_median"] for m, _, _ in points]
    ax.set_xlim(min(xs) / 1.6, max(xs) * 6)
    P.save(fig, config.FIGURES_DIR / "13_clip_speed.png")


# --------------------------------------------------------------------------- run

def run(measure_speed: bool = True) -> dict:
    P.apply_style()
    config.ensure_dirs()
    cnn = load(MAIN)["test"]
    y, source = cnn["y"], cnn["source"]
    probas = {MAIN: cnn["proba"]}
    probe_info = {}
    for name in CLIP_MODELS:
        c = load_clip(name)
        assert (c["test"]["path"] == cnn["path"]).all() and (c["test"]["y"] == y).all(), name
        assert c["classes"] == load(MAIN)["classes"], name
        probas[f"{name}/zero_shot"] = softmax(LOGIT_SCALE * c["test"]["x"] @ c["text"].T)
        logits, probe_info[name] = probe_logits(c)
        probas[f"{name}/probe"] = softmax(logits)
    clip = probas[f"{BEST_CLIP}/zero_shot"]
    confident = cnn["proba"].max(1) >= CONFIDENT
    probas["hybrid"] = np.where(confident[:, None], cnn["proba"], clip)
    probas["average"] = (cnn["proba"] + clip) / 2

    variants = {key: scores(probas[key], y, source) for key in VARIANT_ORDER}
    preds = {key: probas[key].argmax(1) == y for key in VARIANT_ORDER}
    tests = []
    for key in VARIANT_ORDER[1:]:
        for subset in SUBSETS:
            m = source == subset
            tests.append({"a": key, "b": MAIN, "subset": subset, **mcnemar(preds[key][m], preds[MAIN][m])})

    result = {"variants": variants, "significance_vs_cnn": tests, "probe": probe_info,
              "hybrid": {"threshold": CONFIDENT, "cnn_share": round(float(confident.mean()), 4)},
              "prompt": "a photo of {}, a type of food.", "logit_scale": LOGIT_SCALE}
    if measure_speed:
        result["speed"] = speed(CLIP_MODELS)
    write("clip.json", result)
    fig_comparison(variants)
    if measure_speed:
        fig_speed(result["speed"], variants)
    for key in VARIANT_ORDER:
        v = variants[key]
        print(f"{key:22s} food101 {v['food101']['top1']['value']:.3f} / top-3 {v['food101']['top3']['value']:.3f}   "
              f"ua {v['ua']['top1']['value']:.3f} / top-3 {v['ua']['top3']['value']:.3f}")
    return result


if __name__ == "__main__":
    run()
