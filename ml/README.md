# Kolos ML experiments

Training code for the models behind `ml_inference.py`. It lives apart from the web app: the app
never imports this package, it only loads the exported model.

```
ml/
  requirements.txt      training-only dependencies (pinned to the versions of the reported runs)
  plotting.py           one look for every figure
  nutriscore/           Nutri-Score grade (A–E) from nutrition per 100 g
  data/                 datasets and caches      — not in git
  models/               trained weights          — not in git
  reports/nutriscore/   experiment log (README.md, in Ukrainian), figures/, results/*.json
```

## Nutri-Score classifier

Data: the Open Food Facts CSV export (≈1.5 GB gzipped, ODbL licence).

```bash
pip install -r ml/requirements.txt
curl -L -C - -o ml/data/raw/en.openfoodfacts.org.products.csv.gz \
     https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz

python -m ml.nutriscore.extract       # 1. dump → graded products (Parquet)            ~2 min
python -m ml.nutriscore.prepare       # 2. cleaning + fixed train/val/test split      ~20 s
python -m ml.nutriscore.eda           # 3. figures 01–08, results/eda_stats.json      ~1 min
python -m ml.nutriscore.train         # 4. tune, fit and compare the models           the long step
python -m ml.nutriscore.formula       #    official Nutri-Score formula as a reference
python -m ml.nutriscore.experiments   # 5. feature ablation, learning curves
python -m ml.nutriscore.analyze       # 6. figures 09–18, SHAP, calibration, error analysis
python -m ml.nutriscore.export        # 7. compact models for the web app → instance/models/nutriscore/
```

The web app (`score_model.py`) loads the exported models at start-up if they are in
`instance/models/nutriscore/` and LightGBM is installed; otherwise it shows only the grades Open
Food Facts publishes. The exported files are not in git (`instance/` is ignored): run step 7
after cloning, or copy the folder.

Run the commands from the repository root. Every step is deterministic (`config.SEED`); every
number quoted in the report is written to `reports/nutriscore/results/`, every figure to
`reports/nutriscore/figures/`. `python -m ml.nutriscore.train --quick` is a smoke test with two
trials per model.
