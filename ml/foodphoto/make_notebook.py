"""Write the Google Colab notebooks that run the GPU steps on a free GPU:
colab_train.ipynb (steps 2–3, training) and colab_clip.ipynb (step 3b, CLIP embeddings).

The notebooks only drive the scripts of this package, so local and Colab runs share one code
path. Regenerate them after changing the cells below:

    python -m ml.foodphoto.make_notebook
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = "https://github.com/dkrainii/course_project.git"
BRANCH = "phase-f2"
HERE = Path(__file__).parent

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
#    The last line must list 'ua': 879; otherwise read the message above it.
#    Runs in the notebook process, so an error stops "Run all" here.
from pathlib import Path
from ml.foodphoto import colab_data
colab_data.main(Path(DRIVE), workers=4)"""),
]
SETUP = CELLS[1:]  # GPU, Drive, code, data: shared by both notebooks

CELLS += [
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


CLIP_CELLS: list[tuple[str, str]] = [
    ("markdown", """# Kolos: dish recognition — CLIP embeddings on Google Colab

Step 3b of `ml/foodphoto` (see `ml/README.md`): runs CLIP-style image encoders over the same
photos as the trained CNN and saves the vectors to Drive. The comparison itself runs locally
(`python -m ml.foodphoto.clip_analyze`), without a GPU.

**Before the run:** *Runtime → Change runtime type → T4 GPU.* The data cache made by the training
notebook (`prepared.tar` on Drive) is reused, so step 4 takes a few minutes.

**Every session:** *Runtime → Run all.* A finished part is skipped, so after a disconnect just run
everything again.

**At the end:** download the `clip_*`, `siglip2_*`, `dfn5b_h14`, `metaclip2_h14` and
`mobileclip2_s2` folders from `My Drive/Kolos/foodphoto/runs/` and put them into
`ml/models/foodphoto/` locally (~700 MB together)."""),
    *SETUP,
    ("code", """# 5. CLIP library (OpenAI weights are downloaded on first use).
!pip install -q open_clip_torch==4.2.2"""),
    ("code", """# 6. ViT-B/32: the smallest CLIP (~5 min).
!python -m ml.foodphoto.clip_embed --model clip_b32 --workers 2"""),
    ("code", """# 7. ViT-B/16 (~10 min).
!python -m ml.foodphoto.clip_embed --model clip_b16 --workers 2"""),
    ("code", """# 8. ViT-L/14: the largest (~25 min).
!python -m ml.foodphoto.clip_embed --model clip_l14 --workers 2 --batch-size 128"""),
    ("markdown", """## Newer image-text models

Same idea as CLIP, trained on larger and better filtered data. Only the validation and test photos
are encoded (`--splits val test`): enough for the zero-shot and open-vocabulary comparison.
Each model is downloaded once (up to ~4 GB for the H/14 ones)."""),
    ("code", """# 9. SigLIP 2 B/16 (Google), the size of CLIP B/16 (~3 min).
!python -m ml.foodphoto.clip_embed --model siglip2_b16 --workers 2 --splits val test"""),
    ("code", """# 10. SigLIP 2 SO400M/14 (~8 min).
!python -m ml.foodphoto.clip_embed --model siglip2_so400m --workers 2 --batch-size 128 --splits val test"""),
    ("code", """# 11. DFN-5B ViT-H/14 (Apple) (~10 min).
!python -m ml.foodphoto.clip_embed --model dfn5b_h14 --workers 2 --batch-size 128 --splits val test"""),
    ("code", """# 12. MetaCLIP 2 ViT-H/14, worldwide multilingual (Meta) (~10 min).
!python -m ml.foodphoto.clip_embed --model metaclip2_h14 --workers 2 --batch-size 128 --splits val test"""),
    ("code", """# 13. MobileCLIP2-S2 (Apple), a small image encoder (~3 min).
!python -m ml.foodphoto.clip_embed --model mobileclip2_s2 --workers 2 --splits val test"""),
    ("code", """# 14. What is on Drive (text.npz, val.npz, test.npz per model; train.npz for the OpenAI ones).
import glob
for path in sorted(glob.glob(f'{DRIVE}/runs/*/*.npz')):
    if 'predictions' not in path:
        print(f'{os.path.getsize(path) / 1e6:7.1f} MB  {path}')"""),
]


def build(cells_spec: list[tuple[str, str]]) -> dict:
    cells = []
    for kind, source in cells_spec:
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
    for name, spec in (("colab_train.ipynb", CELLS), ("colab_clip.ipynb", CLIP_CELLS)):
        (HERE / name).write_text(json.dumps(build(spec), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {HERE / name}")
