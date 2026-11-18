"""Write colab_train.ipynb: the Google Colab notebook that runs steps 2–3 on a free GPU.

The notebook only drives the scripts of this package, so local and Colab runs share one code
path. Regenerate it after changing the cells below:

    python -m ml.foodphoto.make_notebook
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = "https://github.com/dkrainii/course_project.git"
BRANCH = "phase-f2"
OUT = Path(__file__).with_name("colab_train.ipynb")

CELLS: list[tuple[str, str]] = [
    ("markdown", """# Kolos: dish recognition — training on Google Colab

Runs steps 2–3 of `ml/foodphoto` (see `ml/README.md`) on a free Colab GPU.

**Before the first run**
1. *Runtime → Change runtime type → T4 GPU.*
2. Put `ua_photos.zip` (made locally by `python -m ml.foodphoto.pack`) into
   `My Drive/Kolos/foodphoto/` on Google Drive.

**Every session:** *Runtime → Run all.* Checkpoints are written to Google Drive after each epoch,
so if Colab disconnects, run everything again: training continues from the last finished epoch.

**At the end:** download `My Drive/Kolos/foodphoto/runs/` (without the `*.pt` files, except
`best.pt` of the model that goes into the app) and put it into `ml/models/foodphoto/` locally."""),
    ("code", """# 1. GPU check: a T4 (or better) should be listed.
!nvidia-smi --query-gpu=name,memory.total --format=csv"""),
    ("code", """# 2. Google Drive (persistent storage) and paths.
import os
from google.colab import drive

drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/Kolos/foodphoto'
os.makedirs(f'{DRIVE}/runs', exist_ok=True)
os.environ['FOODPHOTO_DATA'] = '/content/data'          # images: fast local disk
os.environ['FOODPHOTO_RUNS'] = f'{DRIVE}/runs'          # checkpoints and results: Drive
print(sorted(os.listdir(DRIVE)))"""),
    ("code", f"""# 3. Code from GitHub (branch {BRANCH}).
%cd /content
!test -d kolos && git -C kolos pull -q || git clone -q --depth 1 --branch {BRANCH} {REPO} kolos
%cd /content/kolos
!git log --oneline -1"""),
    ("code", """# 4. Data. First session: download Food-101 (~5 GB), resize everything, cache the result on
#    Drive (~2 GB). Later sessions only unpack the cache (~2 min).
import os, subprocess
cache = f'{DRIVE}/prepared.tar'
os.makedirs('/content/data', exist_ok=True)
if os.path.exists(cache):
    !tar -xf "{cache}" -C /content/data
else:
    !unzip -q -o "{DRIVE}/ua_photos.zip" -d /content/data
    !python -m ml.foodphoto.prepare --workers 4
    !tar -cf "{cache}" -C /content/data resized index.csv
!wc -l /content/data/index.csv"""),
    ("markdown", """## Training

Each run takes roughly 40–70 minutes on a T4. Run them in this order; if the session ends,
*Run all* again — finished runs are skipped in seconds and the interrupted one resumes."""),
    ("code", """# 5. Main model: EfficientNet-B0, whole network fine-tuned.
!python -m ml.foodphoto.train --run effb0_finetune --workers 2"""),
    ("code", """# 6. Baseline: EfficientNet-B0 with a frozen backbone (only the classifier is trained).
!python -m ml.foodphoto.train --run effb0_probe --workers 2"""),
    ("code", """# 7. Lighter alternative: MobileNetV3-Large, whole network fine-tuned.
!python -m ml.foodphoto.train --run mnv3_finetune --workers 2"""),
    ("code", """# 8. Ablation: the main model without class-balanced sampling.
!python -m ml.foodphoto.train --run effb0_nobalance --workers 2"""),
    ("code", """# 9. Summary of what is on Drive.
import json, glob
for path in sorted(glob.glob(f'{DRIVE}/runs/*/metrics.json')):
    print(json.load(open(path)))"""),
]


def build() -> dict:
    cells = []
    for kind, source in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
