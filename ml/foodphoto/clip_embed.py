"""Step 3b. CLIP image embeddings of every photo, for the comparison with the trained CNN.

CLIP (Radford et al., 2021) is trained on 400 million image–caption pairs and maps images and text
into one vector space, so it can pick a dish by its *name* without any photos of that dish
("zero-shot"). This step only runs the image encoder over the same train / val / test photos as
train.py (GPU, Colab) and stores the L2-normalised vectors, plus the text vectors of the 108 class
names. Everything else is done locally from these files by clip_analyze.py:

* zero-shot: cosine similarity of a photo to "a photo of {dish}, a type of food.";
* linear probe: a softmax classifier trained on the CLIP vectors of the training photos;
* hybrid: the CNN where it is confident, CLIP elsewhere.

Writes RUNS_DIR/<model>/{train,val,test}.npz (embeddings float16, targets, paths, sources) and
text.npz (class-name vectors). A split that is already written is skipped, so an interrupted
session resumes.

    pip install open_clip_torch
    python -m ml.foodphoto.clip_embed --model clip_b16
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config
from .train import ImageList, class_names, read_index

# name -> (open_clip architecture, pretrained weights). All three are the original OpenAI models.
MODELS = {
    "clip_b32": ("ViT-B-32", "openai"),
    "clip_b16": ("ViT-B-16", "openai"),
    "clip_l14": ("ViT-L-14", "openai"),
}
TEMPLATE = "a photo of {}, a type of food."  # the Food-101 prompt of the CLIP paper
# Class labels as CLIP should read them: Food-101 labels with spaces, Ukrainian dishes with a gloss
# (a transliterated name alone is rare in English captions).
DISPLAY_NAMES = {
    "borscht": "Ukrainian borscht",
    "deruny": "deruny, Ukrainian potato pancakes",
    "holubtsi": "holubtsi, Ukrainian stuffed cabbage rolls",
    "olivier_salad": "Olivier salad",
    "paska": "paska, Ukrainian Easter bread",
    "shashlik": "shashlik",
    "varenyky": "varenyky, Ukrainian dumplings",
}
SPLITS = ("train", "val", "test")


def display_name(label: str) -> str:
    return DISPLAY_NAMES.get(label, label.replace("_", " "))


def load_model(name: str, device: torch.device):
    import open_clip

    arch, weights = MODELS[name]
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=weights, device=device)
    return model.eval(), preprocess, open_clip.get_tokenizer(arch)


@torch.no_grad()
def encode_text(model, tokenizer, texts: list[str], device: torch.device) -> np.ndarray:
    out = []
    for i in range(0, len(texts), 256):
        tokens = tokenizer(texts[i:i + 256]).to(device)
        vec = model.encode_text(tokens).float()
        out.append(torch.nn.functional.normalize(vec, dim=-1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def encode_images(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    out = []
    for i, (images, _) in enumerate(loader):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            vec = model.encode_image(images.to(device, non_blocking=True))
        out.append(torch.nn.functional.normalize(vec.float(), dim=-1).cpu().numpy().astype(np.float16))
        if i % 50 == 0:
            print(f"[clip] {i}/{len(loader)} batches", flush=True)
    return np.concatenate(out)


def embed(name: str, workers: int = 2, batch_size: int = 256, device_name: str | None = None) -> None:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = config.RUNS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    rows = {split: read_index(split) for split in SPLITS}
    classes = class_names(rows["train"])
    for split in ("val", "test"):  # same rule as train.py: only classes that have training photos
        rows[split] = [r for r in rows[split] if r["label"] in classes]
    model, preprocess, tokenizer = load_model(name, device)

    prompts = [TEMPLATE.format(display_name(c)) for c in classes]
    np.savez_compressed(out / "text.npz", classes=np.array(classes), prompts=np.array(prompts),
                        embeddings=encode_text(model, tokenizer, prompts, device).astype(np.float16))
    (out / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=1), encoding="utf-8")

    for split in SPLITS:
        path = out / f"{split}.npz"
        if path.exists():
            print(f"[clip] {name} {split}: already done", flush=True)
            continue
        dataset = ImageList(rows[split], classes, preprocess)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=device.type == "cuda")
        start = time.time()
        vectors = encode_images(model, loader, device)
        print(f"[clip] {name} {split}: {len(dataset)} photos in {time.time() - start:.0f} s", flush=True)
        tmp = out / f"{split}.tmp.npz"  # written fully before it gets its final name
        np.savez_compressed(tmp, embeddings=vectors, targets=np.array(dataset.targets, dtype=np.int16),
                            paths=np.array([r["path"] for r in rows[split]]),
                            sources=np.array([r["source"] for r in rows[split]]))
        tmp.replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device")
    args = parser.parse_args()
    embed(args.model, workers=args.workers, batch_size=args.batch_size, device_name=args.device)
