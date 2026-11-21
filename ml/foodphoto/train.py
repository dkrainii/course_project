"""Step 3. Transfer learning: ImageNet-pretrained CNNs fine-tuned on Food-101 + Ukrainian dishes.

Each run in RUNS is one experiment of the report:

* effb0_probe       EfficientNet-B0, backbone frozen, only the new classifier trained
                    (how far ImageNet features alone go: the baseline);
* effb0_finetune    EfficientNet-B0, whole network fine-tuned (the main model);
* mnv3_finetune     MobileNetV3-Large, whole network fine-tuned (the lighter alternative);
* effb0_nobalance   effb0_finetune without class-balanced sampling (ablation: Ukrainian
                    dishes have ≈10× fewer photos than Food-101 classes).

A run writes to RUNS_DIR/<run>/: last.pt (resumable state, every epoch), best.pt (best
validation top-1), history.json, classes.json and, at the end, predictions.npz (logits of the
best model on validation and test) plus metrics.json. The analysis (analyze.py) and the app
export (export.py) work from those files, so only training needs a GPU.

    python -m ml.foodphoto.train --run effb0_finetune          # resumes if last.pt exists
    python -m ml.foodphoto.train --run effb0_finetune --smoke  # a few batches, any device
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from . import config

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet statistics of the pretrained weights


@dataclass(frozen=True)
class Run:
    arch: str                 # efficientnet_b0 | mobilenet_v3_large
    finetune: bool            # False: only the classifier is trained (linear probe)
    epochs: int
    lr: float                 # classifier; the backbone gets lr * BACKBONE_LR_SCALE
    balance: bool = True      # class-balanced sampling (see sample_weights)
    batch_size: int = 64
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    warmup_epochs: float = 1.0


RUNS = {
    "effb0_probe": Run("efficientnet_b0", finetune=False, epochs=6, lr=3e-3),
    "effb0_finetune": Run("efficientnet_b0", finetune=True, epochs=12, lr=1e-3),
    "mnv3_finetune": Run("mobilenet_v3_large", finetune=True, epochs=12, lr=1e-3),
    "effb0_nobalance": Run("efficientnet_b0", finetune=True, epochs=12, lr=1e-3, balance=False),
}
BACKBONE_LR_SCALE = 0.3
BALANCE_POWER = 0.5   # sample weight ∝ 1 / n_class^power: halfway between natural and uniform


# --------------------------------------------------------------------------- data

def read_index(split: str | None = None) -> list[dict]:
    with open(config.INDEX, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if split is None or r["split"] == split]


def class_names(rows: list[dict]) -> list[str]:
    """Food-101 classes first, then Ukrainian dishes, each sorted: a stable order for every run."""
    food101 = sorted({r["label"] for r in rows if r["source"] == "food101"})
    ua = sorted({r["label"] for r in rows if r["source"] != "food101"} - set(food101))
    return food101 + ua


def train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.35, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(config.RESIZE_TO),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class ImageList(Dataset):
    def __init__(self, rows: list[dict], classes: list[str], transform):
        self.rows, self.transform = rows, transform
        index = {c: i for i, c in enumerate(classes)}
        self.targets = [index[r["label"]] for r in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        with Image.open(config.RESIZED_DIR / self.rows[i]["path"]) as img:
            return self.transform(img.convert("RGB")), self.targets[i]


def sample_weights(targets: list[int], power: float = BALANCE_POWER) -> torch.Tensor:
    counts = np.bincount(targets)
    return torch.as_tensor(1.0 / counts[targets] ** power, dtype=torch.double)


def subsample(rows: list[dict], per_class: int) -> list[dict]:
    seen: dict[str, int] = {}
    out = []
    for r in rows:
        if seen.get(r["label"], 0) < per_class:
            seen[r["label"]] = seen.get(r["label"], 0) + 1
            out.append(r)
    return out


# --------------------------------------------------------------------------- model

def build_model(arch: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
    else:
        raise ValueError(arch)
    last = model.classifier[-1]
    model.classifier[-1] = nn.Linear(last.in_features, n_classes)
    return model


def param_groups(model: nn.Module, run: Run) -> list[dict]:
    head = list(model.classifier[-1].parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    for p in backbone:
        p.requires_grad = run.finetune
    groups = [{"params": head, "lr": run.lr}]
    if run.finetune:
        groups.append({"params": backbone, "lr": run.lr * BACKBONE_LR_SCALE})
    return groups


def lr_factor(step: int, total: int, warmup: int) -> float:
    """Linear warm-up, then cosine decay to zero."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


# --------------------------------------------------------------------------- loops

def seed_everything(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, targets = [], []
    for images, y in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out = model(images.to(device, non_blocking=True).to(memory_format=torch.channels_last))
        logits.append(out.float().cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(logits), np.concatenate(targets)


def topk(logits: np.ndarray, y: np.ndarray, k: int) -> float:
    best = np.argsort(-logits, axis=1)[:, :k]
    return float((best == y[:, None]).any(axis=1).mean())


def train(name: str, smoke: bool = False, workers: int = 2, device_name: str | None = None) -> dict:
    run = RUNS[name]
    seed_everything()
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = config.RUNS_DIR / (name + ("_smoke" if smoke else ""))
    out.mkdir(parents=True, exist_ok=True)

    rows = {split: read_index(split) for split in ("train", "val", "test")}
    classes = class_names(rows["train"])
    for split in ("val", "test"):  # e.g. own photos of a dish that has no training photos
        unknown = sorted({r["label"] for r in rows[split]} - set(classes))
        if unknown:
            print(f"[train] {split}: skipping photos of classes without training data: {unknown}", flush=True)
            rows[split] = [r for r in rows[split] if r["label"] in classes]
    if smoke:
        rows = {s: subsample(r, 4 if s == "train" else 2) for s, r in rows.items()}
    saved = out / "classes.json"
    if (out / "last.pt").exists() and saved.exists() and json.loads(saved.read_text(encoding="utf-8")) != classes:
        raise SystemExit(f"[train] {out} was trained on other classes: delete that folder to start the run again")
    (out / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=1), encoding="utf-8")

    train_set = ImageList(rows["train"], classes, train_transform())
    generator = torch.Generator()
    sampler = (WeightedRandomSampler(sample_weights(train_set.targets), num_samples=len(train_set),
                                     generator=generator) if run.balance else None)
    loader_args = {"batch_size": run.batch_size, "num_workers": workers, "pin_memory": device.type == "cuda",
                   "persistent_workers": workers > 0}
    train_loader = DataLoader(train_set, sampler=sampler, shuffle=sampler is None, drop_last=True, **loader_args)
    val_loader = DataLoader(ImageList(rows["val"], classes, eval_transform()), **loader_args)
    if len(train_loader) == 0:
        train_loader = DataLoader(train_set, shuffle=True, **{**loader_args, "batch_size": len(train_set)})

    model = build_model(run.arch, len(classes), pretrained=True).to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(param_groups(model, run), weight_decay=run.weight_decay)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=run.label_smoothing)
    epochs = 1 if smoke else run.epochs
    total_steps, warmup_steps = epochs * len(train_loader), int(run.warmup_epochs * len(train_loader))

    state = {"epoch": 0, "step": 0, "best_top1": -1.0, "history": []}
    if (out / "last.pt").exists():
        ckpt = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        state = ckpt["state"]
        torch.set_rng_state(ckpt["rng"])
        print(f"[train] {name}: resumed after epoch {state['epoch']}", flush=True)

    print(f"[train] {name}: {len(classes)} classes, {len(train_set):,} train images, device {device}, "
          f"{len(train_loader)} steps/epoch", flush=True)
    for epoch in range(state["epoch"], epochs):
        generator.manual_seed(config.SEED + epoch)  # same sample order per epoch, also after a resume
        model.train()
        if not run.finetune:
            model.features.eval()  # frozen backbone: keep its BatchNorm statistics too
        started, seen, loss_sum = time.perf_counter(), 0, 0.0
        for i, (images, y) in enumerate(train_loader):
            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base * lr_factor(state["step"], total_steps, warmup_steps)
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = criterion(model(images), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            state["step"] += 1
            seen += len(y)
            loss_sum += loss.item() * len(y)
            if i % 200 == 0:
                print(f"[train] epoch {epoch + 1}/{epochs} step {i}/{len(train_loader)} loss {loss.item():.3f}",
                      flush=True)
            if smoke and i >= 2:
                break

        logits, targets = predict(model, val_loader, device)
        record = {
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(seen, 1), 4),
            "val_loss": round(float(nn.functional.cross_entropy(torch.from_numpy(logits),
                                                                torch.from_numpy(targets))), 4),
            "val_top1": round(topk(logits, targets, 1), 4),
            "val_top5": round(topk(logits, targets, 5), 4),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.perf_counter() - started, 1),
        }
        state["history"].append(record)
        state["epoch"] = epoch + 1
        print(f"[train] {name} epoch {epoch + 1}: {record}", flush=True)
        if record["val_top1"] > state["best_top1"]:
            state["best_top1"] = record["val_top1"]
            torch.save({"model": model.state_dict(), "arch": run.arch, "classes": classes, "epoch": epoch + 1},
                       out / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "state": state, "rng": torch.get_rng_state()}, out / "last.pt.tmp")
        (out / "last.pt.tmp").replace(out / "last.pt")  # a dropped session never leaves a broken file
        (out / "history.json").write_text(json.dumps({"run": name, "config": asdict(run), "history": state["history"]},
                                                     indent=1), encoding="utf-8")

    return finish(name, out, rows, classes, device, loader_args)


def finish(name: str, out: Path, rows: dict, classes: list[str], device: torch.device, loader_args: dict) -> dict:
    """Logits of the best checkpoint on validation and test, for analyze.py and export.py."""
    best = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model = build_model(best["arch"], len(classes), pretrained=False)
    model.load_state_dict(best["model"])
    model = model.to(device).to(memory_format=torch.channels_last)
    arrays, metrics = {}, {"run": name, "best_epoch": best["epoch"], "classes": len(classes)}
    for split in ("val", "test"):
        logits, targets = predict(model, DataLoader(ImageList(rows[split], classes, eval_transform()), **loader_args),
                                  device)
        arrays[f"{split}_logits"] = logits.astype(np.float16)
        arrays[f"{split}_targets"] = targets.astype(np.int16)
        arrays[f"{split}_sources"] = np.array([r["source"] for r in rows[split]])
        arrays[f"{split}_paths"] = np.array([r["path"] for r in rows[split]])
        metrics[split] = {"top1": round(topk(logits, targets, 1), 4), "top5": round(topk(logits, targets, 5), 4)}
    np.savez_compressed(out / "predictions.npz", **arrays)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(f"[train] {name} done: {metrics}", flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", choices=sorted(RUNS), required=True)
    parser.add_argument("--smoke", action="store_true", help="a few batches on a tiny subset")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device")
    args = parser.parse_args()
    train(args.run, smoke=args.smoke, workers=args.workers, device_name=args.device)
