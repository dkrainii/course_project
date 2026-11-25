"""The trained dish-recognition model inside the web app.

`ml/foodphoto/export.py` writes model.onnx (EfficientNet-B0 with a softmax at the end) and
meta.json (class order, preprocessing, confidence threshold) to `instance/models/foodphoto/`.
This module loads them with onnxruntime and registers one callable with `ml_inference`:
"photo bytes, k → the k most likely dishes with their probabilities".

The preprocessing repeats the evaluation transform of training with PIL and numpy (no PyTorch
in the web app): turn by EXIF, RGB, shorter side to 256 px, centre crop 224 × 224, ImageNet
mean/std. export.py checks that both give the same answers.

onnxruntime is imported lazily: without it, or without the exported files, the app runs as
before and the photo sheet says recognition is not available yet.
"""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import ml_inference

logger = logging.getLogger(__name__)

THREADS = 2  # per request; the server answers several users at once


class FoodPhotoModel:
    def __init__(self, directory: Path):
        import onnxruntime as ort  # optional dependency, see requirements.txt

        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        self.classes: list[str] = meta["classes"]
        spec = meta["input"]
        self.resize, self.crop = int(spec["resize"]), int(spec["crop"])
        self.mean = np.array(spec["mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(spec["std"], dtype=np.float32).reshape(3, 1, 1)
        self.confident = float(meta.get("confident", 1.0))
        options = ort.SessionOptions()
        options.intra_op_num_threads = THREADS
        self.session = ort.InferenceSession(str(directory / meta["file"]), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        n_out = self.session.get_outputs()[0].shape[-1]
        if n_out != len(self.classes):
            raise ValueError(f"model has {n_out} outputs but meta.json lists {len(self.classes)} classes")

    def preprocess(self, image: bytes) -> np.ndarray:
        try:
            return self._preprocess(image)
        except (OSError, SyntaxError, Image.DecompressionBombError) as exc:  # not an image / broken / huge
            raise ValueError(f"cannot read the photo: {exc}") from exc

    def _preprocess(self, image: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(image)) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            w, h = img.size
            # torchvision.transforms.Resize(int): shorter side → resize, the other side truncated
            if w <= h:
                size = (self.resize, int(self.resize * h / w))
            else:
                size = (int(self.resize * w / h), self.resize)
            img = img.resize(size, Image.BILINEAR)
            w, h = img.size
            left, top = int(round((w - self.crop) / 2.0)), int(round((h - self.crop) / 2.0))
            img = img.crop((left, top, left + self.crop, top + self.crop))
            pixels = np.asarray(img, dtype=np.float32) / 255.0
        return ((pixels.transpose(2, 0, 1) - self.mean) / self.std)[None]

    def probabilities(self, image: bytes) -> np.ndarray:
        return self.session.run(None, {self.input_name: self.preprocess(image)})[0][0]

    def __call__(self, image: bytes, top_k: int) -> list[tuple[str, float]]:
        proba = self.probabilities(image)
        best = np.argsort(-proba)[:top_k]
        return [(self.classes[i], float(proba[i])) for i in best]


def default_directory(app) -> Path:
    return Path(app.instance_path) / "models" / "foodphoto"


def load_photo_model(app) -> bool:
    """Register the exported model if it is there. Config PHOTO_MODEL_DIR: a path, or None to skip."""
    directory = app.config.get("PHOTO_MODEL_DIR", default_directory(app))
    if not directory:
        return False
    directory = Path(directory)
    if not (directory / "meta.json").is_file():
        logger.info("Dish photo model not found in %s; photo recognition is off.", directory)
        return False
    try:
        model = FoodPhotoModel(directory)
    except ImportError:
        logger.warning("onnxruntime is not installed; the dish photo model in %s is not used.", directory)
        return False
    except (OSError, KeyError, ValueError) as exc:
        logger.warning("Could not load the dish photo model from %s: %s", directory, exc)
        return False
    ml_inference.register_photo_model(model)
    logger.info("Dish photo model loaded from %s (%d classes).", directory, len(model.classes))
    return True
