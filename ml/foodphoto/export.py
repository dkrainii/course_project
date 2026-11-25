"""Step 5. The main model for the web app: best.pt → ONNX.

Writes to instance/models/foodphoto/ (not in git):

* model.onnx — EfficientNet-B0 with softmax at the end: float32 [N, 3, 224, 224] in, class
  probabilities [N, 108] out. The web app runs it with onnxruntime (≈15 MB, no PyTorch);
* meta.json  — class order, preprocessing (resize, crop, mean, std), the confidence threshold and
  the test metrics, so photo_model.py needs nothing from this package.

The export is checked twice: ONNX against PyTorch on the same tensors, and the app's own
preprocessing (photo_model.preprocess, PIL + numpy) against torchvision's on real photos.

    python -m ml.foodphoto.export            # → instance/models/foodphoto/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from . import config
from .train import MEAN, STD, build_model, eval_transform

ROOT = config.ML_ROOT.parent
DEFAULT_OUT = ROOT / "instance" / "models" / "foodphoto"
RUN = "effb0_finetune"
# Interface rule from the report (section 7): from this confidence on, the first suggestion is
# right in ≈94 % of cases and is shown as the likely answer.
CONFIDENT = 0.5
OPSET = 17


class WithSoftmax(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(x), dim=1)


def load(run: str = RUN) -> tuple[nn.Module, list[str]]:
    ckpt = torch.load(config.RUNS_DIR / run / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["model"])
    return WithSoftmax(model).eval(), ckpt["classes"]


def sample_photos(limit: int = 16) -> list[Path]:
    photos = sorted(config.UA_DIR.glob("*/*.jpg"))
    return photos[:: max(1, len(photos) // limit)][:limit]


def export(out: Path = DEFAULT_OUT, run: str = RUN) -> dict:
    import onnxruntime as ort

    import photo_model  # the web app's loader: its preprocessing is checked here

    model, classes = load(run)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "model.onnx"
    dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    torch.onnx.export(model, dummy, str(onnx_path), input_names=["image"], output_names=["probabilities"],
                      dynamic_axes={"image": {0: "batch"}, "probabilities": {0: "batch"}},
                      opset_version=OPSET, dynamo=False)

    metrics = json.loads((config.RUNS_DIR / run / "metrics.json").read_text(encoding="utf-8"))
    meta = {
        "run": run,
        "arch": "efficientnet_b0",
        "file": onnx_path.name,
        "classes": classes,
        "input": {"resize": config.RESIZE_TO, "crop": config.IMAGE_SIZE, "mean": list(MEAN), "std": list(STD)},
        "confident": CONFIDENT,
        "test": metrics["test"],
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # 1. ONNX = PyTorch on random input and on real photos (torchvision preprocessing).
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    photos = sample_photos()
    transform = eval_transform()
    batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in photos])
    with torch.inference_mode():
        expected = model(batch).numpy()
    got = session.run(None, {"image": batch.numpy()})[0]
    max_diff = float(np.abs(expected - got).max())
    same_top1 = bool((expected.argmax(1) == got.argmax(1)).all())

    # 2. The app's preprocessing (PIL + numpy, no torchvision) gives the same answers.
    loaded = photo_model.FoodPhotoModel(out)
    app_top1 = [loaded(p.read_bytes(), 1)[0][0] for p in photos]
    torch_top1 = [classes[i] for i in expected.argmax(1)]
    app_agrees = sum(a == b for a, b in zip(app_top1, torch_top1))

    report = {"onnx_mb": round(onnx_path.stat().st_size / 2**20, 1), "photos": len(photos),
              "max_abs_diff": max_diff, "same_top1": same_top1, "app_preprocessing_agrees": app_agrees}
    print(f"[export] {onnx_path} ({report['onnx_mb']} MB): max |ONNX − PyTorch| = {max_diff:.2e}, "
          f"top-1 identical: {same_top1}; app preprocessing agrees on {app_agrees}/{len(photos)} photos", flush=True)
    if max_diff > 1e-3 or not same_top1:
        raise SystemExit("[export] ONNX output differs from PyTorch")
    if app_agrees < len(photos) - 1:  # resampling differs a little between PIL and torchvision
        raise SystemExit("[export] the app's preprocessing does not match training")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run", default=RUN)
    args = parser.parse_args()
    export(args.out, args.run)
