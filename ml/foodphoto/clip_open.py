"""Step 4c. Open vocabulary: CLIP picks the dish among all 5,432 foods of USDA FNDDS, not 108 names.

The trained CNN can only answer with one of its 108 classes. CLIP compares a photo with any text,
so the list of dishes can be the whole food database the app takes its nutrition data from. This
step measures how often CLIP, choosing among every FNDDS description ("Pizza, cheese, from
restaurant or fast food, thin crust", "Soup, borscht", "Pierogi", ... and also milk, raw
vegetables, baby food), lands on a record of the right dish.

Two vocabularies are scored: every FNDDS record, and only the dishes (no drinks, sauces,
condiments, baby food or "as ingredient" records; see NOT_DISH_*), which is what the app would
offer. A test photo counts as recognised when the chosen record matches the regular expression of its
class in ACCEPT: any variant of the same dish is right (thin or thick crust pizza, beef or chicken
taco), a different dish is wrong. Classes without a real counterpart in FNDDS (risotto, takoyaki,
cheese plate, paska, ...) are listed in NO_MATCH and left out of this test. The same photos are
also scored with the 108 class names (the closed set of clip_analyze.py), so the cost of the large
vocabulary is measured on identical images.

Text vectors are cached in RUNS_DIR/<model>/fndds_text.npz (CPU, a few minutes for L/14).

    python -m ml.foodphoto.clip_open   → results/clip_open.json, figures/14_clip_open.png
"""
from __future__ import annotations

import csv
import re
from collections import Counter

import numpy as np
import torch

from .. import plotting as P
from . import config
from .analyze import DISHES_UK, MAIN, SUBSETS, accuracy, load, mcnemar, num, write
from .clip_analyze import CLIP_LABELS, CLIP_MODELS, load_clip
from .clip_embed import MODELS, TEMPLATE

plt = P.plt

FNDDS_FOOD = config.ML_ROOT / "data" / "raw" / "fndds" / "FoodData_Central_survey_food_csv_2024-10-31" / "food.csv"
TOP_K = (1, 3, 5)

# Which FNDDS descriptions are the same dish as a class (re.search, case-insensitive).
ACCEPT = {
    "apple_pie": r"^Pie, apple",
    "baby_back_ribs": r"^Pork, ribs|^Ribs, NFS",
    "baklava": r"^Baklava",
    "beef_tartare": r"^Steak tartare",
    "beet_salad": r"^Beets,",
    "beignets": r"^Beignet",
    "bibimbap": r"^Bibimbap",
    "bread_pudding": r"^Pudding, bread",
    "breakfast_burrito": r"^Egg burrito",
    "bruschetta": r"^Bruschetta",
    "caesar_salad": r"caesar garden salad|^Caesar salad",
    "cannoli": r"^Pastry, Italian, with cheese",
    "caprese_salad": r"^Mozzarella cheese, tomato, and basil",
    "carrot_cake": r"^Cake or cupcake, carrot",
    "ceviche": r"^Ceviche",
    "cheesecake": r"^Cheesecake",
    "chicken_curry": r"^Chicken curry",
    "chicken_quesadilla": r"^Quesadilla",
    "chicken_wings": r'^Chicken wing|^Chicken "wings"',
    "chocolate_cake": r"^Cake or cupcake, (chocolate|German chocolate|Black Forest)|^Cake, chocolate",
    "chocolate_mousse": r"^Mousse$",
    "churros": r"^Churros",
    "clam_chowder": r"clam chowder",
    "club_sandwich": r"^Club sandwich",
    "crab_cakes": r"^Crab, cake",
    "creme_brulee": r"^Creme brulee",
    "cup_cakes": r"^Cake or cupcake",
    "deviled_eggs": r"^Egg, deviled",
    "donuts": r"^Doughnut",
    "dumplings": r"^Wonton, dumpling or pot sticker|^Dumpling, no meat",
    "edamame": r"^Edamame",
    "eggs_benedict": r"^Egg, Benedict",
    "escargots": r"^Escargot",
    "falafel": r"^Falafel",
    "filet_mignon": r"^Beef, steak, tenderloin",
    "fish_and_chips": r"^Fish, [^,]+, fried$|^Fish, NS as to type, fried",
    "foie_gras": r"^Liver, paste or pate",
    "french_fries": r"^Potato, french fries",
    "french_onion_soup": r"^Soup, French onion",
    "french_toast": r"^French toast",
    "fried_calamari": r"^Calamari",
    "fried_rice": r"^Rice, fried",
    "frozen_yogurt": r"^Frozen yogurt(?! (sandwich|bar))",
    "garlic_bread": r"^Garlic bread",
    "gnocchi": r"^Gnocchi",
    "greek_salad": r"^Greek Salad",
    "grilled_cheese_sandwich": r"^Grilled cheese sandwich|^Cheese sandwich",
    "grilled_salmon": r"^Fish, salmon, (grilled|baked or broiled)",
    "guacamole": r"^Guacamole",
    "gyoza": r"^Wonton, dumpling or pot sticker",
    "hamburger": r"^(Double )?(Hamburger|Cheeseburger)|^Whopper",
    "hot_and_sour_soup": r"^Soup, hot and sour",
    "hot_dog": r"^Hot dog|^Chili hot dog",
    "huevos_rancheros": r"^Huevos rancheros",
    "hummus": r"^Hummus",
    "ice_cream": r"^(Light )?Ice cream(, (?!fried)| cone| sundae)|^Gelato",
    "lasagna": r"^Lasagna",
    "lobster_bisque": r"^Soup, bisque",
    "lobster_roll_sandwich": r"^Lobster salad",
    "macaroni_and_cheese": r"^Macaroni or noodles,? (creamed, )?with cheese",
    "macarons": r"^Cookie, macaroon",
    "miso_soup": r"^Soup, miso",
    "mussels": r"^Mussels",
    "nachos": r"^Nachos",
    "omelette": r"^Egg omelet",
    "onion_rings": r"onion rings$",
    "oysters": r"^Oysters",
    "pad_thai": r"^Pad Thai",
    "paella": r"paella",
    "pancakes": r"^Pancakes",
    "peking_duck": r"^Duck, (Peking|roasted)",
    "pho": r"^Soup, pho",
    "pizza": r"^Pizza",
    "pork_chop": r"^Pork, chop",
    "prime_rib": r"^Beef, roast$|^Beef, steak, ribeye",
    "pulled_pork_sandwich": r"^Barbecue sandwich|pulled pork",
    "ramen": r"^Ramen bowl|^Soup, ramen",
    "ravioli": r"^Ravioli",
    "red_velvet_cake": r"red velvet",
    "samosa": r"^Samosa",
    "sashimi": r"^Fish, (tuna, |salmon, )?raw",
    "scallops": r"^Scallops",
    "seaweed_salad": r"^Seaweed",
    "spaghetti_bolognese": r"^Pasta with tomato-based sauce and meat|^Spaghetti with meat sauce",
    "spring_rolls": r"^Egg roll|spring roll|^Roll with meat and/or shrimp, vegetables and rice paper",
    "steak": r"^Beef, steak|^Steak, ",
    "strawberry_shortcake": r"strawberry shortcake",
    "sushi": r"^Sushi",
    "tacos": r"^Taco, ",
    "tiramisu": r"^Tiramisu",
    "tuna_tartare": r"^Fish, tuna, raw",
    "waffles": r"^Waffle, ",
    "borscht": r"borscht",
    "deruny": r"^Potato pancake",
    "holubtsi": r"^Stuffed cabbage",
    "olivier_salad": r"^Potato salad with egg",
    "shashlik": r"shish kabob",
    "varenyky": r"^Pierogi",
}
# FNDDS has only a loose stand-in for these (the catalogue uses one, see build_usda_micros.FNDDS_IDS).
NO_MATCH = ["beef_carpaccio", "cheese_plate", "croque_madame", "panna_cotta", "poutine", "risotto",
            "shrimp_and_grits", "spaghetti_carbonara", "takoyaki", "paska"]


# The "dishes" vocabulary leaves out what nobody photographs as a meal: drinks, fats and dressings,
# sugars and syrups, condiments, baby food and formula, dry mixes (WWEIA categories), and sauces or
# ingredients of other foods (by name). Olives, pickles and dips (hummus, guacamole) stay.
NOT_DISH_CATEGORIES = [(1002, 1402), (1902, 1904), (7002, 7804), (8002, 8012), (8402, 8406), (8410, 8410),
                       (8802, 8806), (9002, 9999)]
# A sauce or gravy named before any "with" is the food itself ("Gravy, beef", "Alfredo sauce with
# seafood"); after "with" it is part of a dish ("Pasta with tomato-based sauce and meat").
NOT_DISH_NAME = re.compile(r"^(?:(?!\bwith\b)[^,])*\b(?:sauce|gravy)\b|as ingredient|^Topping from|for use (?:with|on)",
                           re.I)


def fndds_foods() -> tuple[list[str], np.ndarray, np.ndarray]:
    """FNDDS descriptions, a mask of those that are dishes (see NOT_DISH_*) and kcal per 100 g."""
    with open(FNDDS_FOOD, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    kcal = {}
    with open(FNDDS_FOOD.with_name("food_nutrient.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["nutrient_id"] == "208":  # FNDDS files give the nutrient number: 208 = energy, kcal
                kcal[r["fdc_id"]] = float(r["amount"])
    dish = [not any(lo <= int(r["food_category_id"]) <= hi for lo, hi in NOT_DISH_CATEGORIES)
            and not NOT_DISH_NAME.search(r["description"]) and r["fdc_id"] in kcal for r in rows]
    return ([r["description"] for r in rows], np.array(dish),
            np.array([kcal.get(r["fdc_id"], np.nan) for r in rows]))


def catalogue_kcal(classes: list[str]) -> np.ndarray:
    """kcal per 100 g of the catalogue food each class is logged as (seed_foods.PHOTO_FOODS)."""
    import sys

    sys.path.insert(0, str(config.ML_ROOT.parent))
    from seed_foods import GENERIC_FOODS, PHOTO_FOODS

    kcal = {row[0]: row[1] for row in GENERIC_FOODS}
    return np.array([kcal[PHOTO_FOODS[c]] for c in classes], dtype=float)


def kcal_error(predicted: np.ndarray, true: np.ndarray) -> dict:
    """How far the kcal per 100 g of the chosen food is from the right dish's (wrong answers included)."""
    err = np.abs(predicted - true)
    return {"median_abs": round(float(np.median(err)), 1), "mean_abs": round(float(err.mean()), 1),
            "within_20pct": accuracy(err <= 0.2 * true)}


def fndds_text(name: str, names: list[str]) -> np.ndarray:
    """Vectors of TEMPLATE.format(description) for every FNDDS food, cached per model."""
    path = config.RUNS_DIR / name / "fndds_text.npz"
    if path.exists():
        cached = np.load(path)
        if [str(n) for n in cached["names"]] == names:
            return cached["embeddings"].astype(np.float32)
    import open_clip

    from .clip_embed import encode_text

    arch, weights = MODELS[name]
    torch.set_num_threads(4)
    model = open_clip.create_model(arch, pretrained=weights).eval()
    model.visual = None  # only the text encoder is needed here
    vectors = encode_text(model, open_clip.get_tokenizer(arch), [TEMPLATE.format(n) for n in names],
                          torch.device("cpu"))
    np.savez_compressed(path, names=np.array(names), embeddings=vectors.astype(np.float16))
    print(f"[clip] {name}: {len(names)} FNDDS names encoded", flush=True)
    return vectors


def accept_matrix(classes: list[str], names: list[str]) -> np.ndarray:
    """accept[c, j]: FNDDS record j is the dish of class c."""
    out = np.zeros((len(classes), len(names)), bool)
    for c, label in enumerate(classes):
        if label in ACCEPT:
            pattern = re.compile(ACCEPT[label], re.I)
            out[c] = [bool(pattern.search(n)) for n in names]
            assert out[c].any(), f"{label}: no FNDDS record matches {ACCEPT[label]!r}"
    return out


# FNDDS records allowed, plus class names of our catalogue added as extra entries (the prompts of
# clip_embed: Food-101 labels, Ukrainian dishes with a gloss). An extra entry is right for its class.
VOCABULARIES = ["all", "dishes", "dishes_ua", "dishes_catalogue"]


def evaluate(name: str, names: list[str], dish: np.ndarray, kcal: np.ndarray) -> dict:
    c = load_clip(name)
    classes, test = c["classes"], c["test"]
    assert set(ACCEPT) | set(NO_MATCH) == set(classes), set(classes) ^ (set(ACCEPT) | set(NO_MATCH))
    accept = accept_matrix(classes, names)
    assert (accept & dish).any(axis=1)[[classes.index(k) for k in ACCEPT]].all(), "a class lost all its records"
    keep = np.isin(np.array(classes)[test["y"]], list(ACCEPT))
    x, y, source = test["x"][keep], test["y"][keep], test["source"][keep]
    cat_kcal = catalogue_kcal(classes)

    sims, class_sims = x @ fndds_text(name, names).T, x @ c["text"].T
    ua = [i for i, label in enumerate(classes) if label in DISHES_UK]
    setups = {"all": (np.ones(len(names), bool), []), "dishes": (dish, []),
              "dishes_ua": (dish, ua), "dishes_catalogue": (dish, list(range(len(classes))))}
    hits, tops, entry_kcal, entry_names = {}, {}, {}, {}
    for vocab, (allowed, extra) in setups.items():
        s = np.hstack([np.where(allowed, sims, -np.inf), class_sims[:, extra]])
        right = np.hstack([accept, np.eye(len(classes), dtype=bool)[:, extra]])
        tops[vocab] = np.argsort(-s, axis=1)[:, :max(TOP_K)]
        hits[vocab] = right[y[:, None], tops[vocab]]         # [photo, rank]: is the k-th entry right?
        entry_kcal[vocab] = np.concatenate([kcal, cat_kcal[extra]])
        entry_names[vocab] = names + [c["prompts"][i] for i in extra]
    closed_rank = np.argsort(-(x @ c["text"].T), axis=1)[:, :max(TOP_K)]
    closed, closed_pred = closed_rank == y[:, None], closed_rank[:, 0]
    cnn = load(MAIN)["test"]
    assert (cnn["path"] == test["path"]).all()
    cnn_pred = cnn["pred"][keep]

    out = {"photos": int(keep.sum()), "classes": len(ACCEPT),
           "vocabulary": {v: int(setups[v][0].sum()) + len(setups[v][1]) for v in VOCABULARIES}}
    for subset in ("all", *SUBSETS):
        m = np.ones(len(y), bool) if subset == "all" else source == subset
        out[subset] = {f"closed_top{k}": accuracy(closed[m, :k].any(1)) for k in TOP_K}
        for vocab, hit in hits.items():
            out[subset].update({f"{vocab}_top{k}": accuracy(hit[m, :k].any(1)) for k in TOP_K})
        # kcal per 100 g of the first answer against the catalogue food of the true class
        out[subset]["kcal_error"] = {
            "cnn": kcal_error(cat_kcal[cnn_pred[m]], cat_kcal[y[m]]),
            "closed": kcal_error(cat_kcal[closed_pred[m]], cat_kcal[y[m]]),
            **{v: kcal_error(entry_kcal[v][tops[v][m, 0]], cat_kcal[y[m]]) for v in VOCABULARIES[1:]},
        }
    # do better names help? the same photos, top-1 with and without the extra entries (exact McNemar)
    out["significance"] = [{"a": v, "b": "dishes", "subset": subset,
                            **mcnemar(hits[v][source == subset, 0], hits["dishes"][source == subset, 0])}
                           for v in ("dishes_ua", "dishes_catalogue") for subset in SUBSETS]
    per_class = []
    top, hit = tops["dishes"], hits["dishes"]
    for ci, label in enumerate(classes):
        m = y == ci
        if not m.any():
            continue
        wrong = Counter(names[j] for j, ok in zip(top[m, 0], hit[m, 0]) if not ok)
        right = Counter(names[j] for j, ok in zip(top[m, 0], hit[m, 0]) if ok)
        per_class.append({"class": label, "n": int(m.sum()), "closed_top1": round(float(closed[m, 0].mean()), 4),
                          "all_top1": round(float(hits["all"][m, 0].mean()), 4),
                          "dishes_top1": round(float(hit[m, 0].mean()), 4),
                          "dishes_top3": round(float(hit[m, :3].any(1).mean()), 4),
                          "dishes_ua_top1": round(float(hits["dishes_ua"][m, 0].mean()), 4),
                          "dishes_catalogue_top1": round(float(hits["dishes_catalogue"][m, 0].mean()), 4),
                          "catalogue_mistake": Counter(entry_names["dishes_catalogue"][j] for j, ok in zip(
                              tops["dishes_catalogue"][m, 0], hits["dishes_catalogue"][m, 0]) if not ok
                          ).most_common(1)[0][0] if not hits["dishes_catalogue"][m, 0].all() else None,
                          "accepted_records": int((accept[ci] & dish).sum()),
                          "usual_answer": right.most_common(1)[0][0] if right else None,
                          "usual_mistake": wrong.most_common(1)[0][0] if wrong else None})
    out["per_class"] = sorted(per_class, key=lambda r: r["dishes_top1"])
    return out


def fig_open(results: dict) -> None:
    n_dishes = P.thousands(results[CLIP_MODELS[0]]["vocabulary"]["dishes"])
    bars = [("closed_top1", P.NEUTRAL, "лише 108 назв класів"),
            ("dishes_top1", P.ORANGE, f"{n_dishes} страв FNDDS (назви USDA)"),
            ("dishes_ua_top1", P.VIOLET, "+ 7 українських страв зі зрозумілими назвами"),
            ("dishes_catalogue_top1", P.AQUA, "+ усі 108 страв каталогу зі зрозумілими назвами")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, subset in zip(axes, SUBSETS):
        for i, name in enumerate(CLIP_MODELS):
            r = results[name][subset]
            for j, (key, color, label) in enumerate(bars):
                m = r[key]
                y = i + (j - 1.5) * 0.21
                value, (low, high) = m["value"] * 100, (v * 100 for v in m["ci95"])
                ax.barh(y, value, height=0.19, color=color, label=label if i == 0 else None)
                ax.errorbar(value, y, xerr=[[value - low], [high - value]], color=P.INK, capsize=2, linewidth=0.8)
                ax.text(high + 0.8, y, num(value), va="center", fontsize=7.5)
        P.horizontal_grid(ax)
        ax.set_xlim(0, 110)
        ax.set_title(f"{SUBSETS[subset]} (n = {P.thousands(results[CLIP_MODELS[0]][subset]['closed_top1']['n'])})")
        ax.set_xlabel("Перша відповідь — правильна страва, % тестових фото")
    axes[0].set_yticks(range(len(CLIP_MODELS)), [CLIP_LABELS[n] for n in CLIP_MODELS])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8.5, frameon=False, title="Словник:",
               title_fontsize=8.5)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    P.save(fig, config.FIGURES_DIR / "14_clip_open.png")


def run() -> dict:
    P.apply_style()
    config.ensure_dirs()
    names, dish, kcal = fndds_foods()
    results = {name: evaluate(name, names, dish, kcal) for name in CLIP_MODELS}
    results["no_match_classes"] = NO_MATCH
    results["accept"] = ACCEPT
    results["prompt"] = TEMPLATE
    write("clip_open.json", results)
    fig_open(results)
    for name in CLIP_MODELS:
        r = results[name]
        print(f"{name} {r['vocabulary']}")
        for s in SUBSETS:
            print(f"  {s:8s} closed {r[s]['closed_top1']['value']:.3f}  " + "  ".join(
                f"{v} {r[s][v + '_top1']['value']:.3f}/{r[s][v + '_top3']['value']:.3f}" for v in VOCABULARIES)
                + "  kcal±20%: " + " ".join(f"{k} {e['within_20pct']['value']:.2f}" for k, e in r[s]["kcal_error"].items()))
    for row in results["clip_l14"]["per_class"]:
        if row["class"] in DISHES_UK:
            print(f"  {row['class']:14s} dishes {row['dishes_top1']:.2f} +ua {row['dishes_ua_top1']:.2f} "
                  f"+catalogue {row['dishes_catalogue_top1']:.2f}  mistake: {row['catalogue_mistake']}")
    return results


if __name__ == "__main__":
    run()
