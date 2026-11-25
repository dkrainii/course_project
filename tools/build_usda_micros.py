"""Build usda_micros.py: micronutrients and label fields for the built-in foods from USDA FoodData Central.

Sources, both public domain (CC0):
* SR Legacy (April 2018) — foods and ingredients:
    https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_csv_2018-04.zip
* FNDDS (Survey Foods, October 2024) — prepared dishes, for the dishes the photo model
  recognises (seed_foods.PHOTO_FOODS):
    https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_survey_food_csv_2024-10-31.zip

Usage (from the project root, with both zips extracted anywhere):
    python tools/build_usda_micros.py path/to/FoodData_Central_sr_legacy_food_csv_2018-04 \
                                      path/to/FoodData_Central_survey_food_csv_2024-10-31

Every built-in food (app.PRODUCT_SEED + seed_foods.GENERIC_FOODS) is mapped by hand to one USDA
record (FDC_IDS or FNDDS_IDS), picked by description and checked against our kcal per 100 g. The
macros of the dishes in seed_foods come from the same FNDDS record, and the build checks that
they match. Foods USDA does not describe were dropped from the catalogue
(seed_foods.RETIRED_FOODS) rather than guessed.

The label fields (sugars, saturated fat, fibre, salt) feed the Nutri-Score model, which needs them
all, so the few that SR Legacy leaves out are filled from LABEL_ESTIMATES: a close USDA record
scaled to our carbs, or the median of the same food in Open Food Facts. Each estimate names its
source and is listed in usda_micros.LABEL_ESTIMATED.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "usda_micros.py"

# Our nutrient name → SR Legacy nutrient id (units match app.MICRONUTRIENT_CATALOG).
NUTRIENT_IDS = {
    "Potassium": 1092,    # mg
    "Sodium": 1093,       # mg
    "Magnesium": 1090,    # mg
    "Calcium": 1087,      # mg
    "Zinc": 1095,         # mg
    "Iron": 1089,         # mg
    "Vitamin C": 1162,    # mg
    "Vitamin D": 1110,    # IU
    "Vitamin B12": 1178,  # µg
}
VITAMIN_D_UG = 1114       # used when only µg is reported (1 µg = 40 IU)
# Omega-3 (g) = ALA + EPA + DPA + DHA; ALA is often reported only as undifferentiated 18:3.
ALA, ALA_18_3, EPA, DPA, DHA = 1404, 1270, 1278, 1280, 1272

# Label field → SR Legacy nutrient id (g per 100 g). Salt = sodium (mg) / 1000 × 2.5.
LABEL_IDS = {"sugars": 2000, "saturated_fat": 1258, "fiber": 1079}
SODIUM = 1093
SALT_PER_SODIUM = 2.5

# Values SR Legacy does not report for our record: field → (value per 100 g, source).
LABEL_ESTIMATES: dict[str, dict[str, tuple[float, str]]] = {
    "French fries": {"sugars": (0.4, "USDA 170118 (oven-heated fries), scaled to our carbs")},
    "Pancakes": {"sugars": (6.2, "USDA 172771 (plain pancakes, frozen), scaled to our carbs"),
                 "fiber": (0.8, "USDA 172771 (plain pancakes, frozen), scaled to our carbs")},
    "Mackerel, cooked": {"sugars": (0.0, "USDA 175119 (Atlantic mackerel, raw): no carbohydrate")},
    "Shrimp, cooked": {"sugars": (0.0, "USDA 171971 (shrimp, cooked, moist heat)"),
                       "fiber": (0.0, "USDA 171971 (shrimp, cooked, moist heat)")},
    "Tofu, firm": {"sugars": (0.6, "USDA 172448 (firm tofu, nigari), scaled to our carbs")},
    "Tempeh": {"sugars": (0.7, "Open Food Facts: median of 126 tempeh products"),
               "fiber": (6.5, "Open Food Facts: median of 126 tempeh products")},
    "Chia seeds": {"sugars": (0.0, "Open Food Facts: median of 194 chia seed products")},
}

FDC_IDS: dict[str, int] = {
    # app.PRODUCT_SEED
    "Chicken breast": 171477, "Brown rice": 169704, "Greek yogurt": 171304, "Banana": 173944,
    "Oatmeal": 173905,
    # Grains, bread & starches
    "White rice, cooked": 168878, "Basmati rice, cooked": 169708, "Buckwheat, cooked": 170686,
    "Quinoa, cooked": 168917, "Pasta, cooked": 169737, "Whole wheat pasta, cooked": 168910,
    "Bulgur, cooked": 170287, "Couscous, cooked": 169700, "Millet, cooked": 168871,
    "Pearl barley, cooked": 170285, "Rolled oats, dry": 173904, "Cornflakes": 174648,
    "Granola": 171646, "White bread": 174924, "Whole wheat bread": 172688, "Rye bread": 172684,
    "Flour tortilla": 175037, "Pita bread": 174915, "Bagel": 174899, "Rice cakes": 170250,
    "Potato, boiled": 170438, "Sweet potato, baked": 168483, "French fries": 169268,
    "Mashed potatoes": 168555, "Sweet corn, boiled": 169999, "Pancakes": 175009, "Croissant": 174987,
    # Meat & poultry
    "Chicken thigh, skinless, cooked": 172388, "Chicken wings, roasted": 173630,
    "Turkey breast, roasted": 174516, "Ground beef 90% lean, cooked": 174031,
    "Beef sirloin steak, grilled": 172171, "Beef liver, cooked": 168626, "Pork loin, roasted": 167842,
    "Pork chop, cooked": 168298, "Ham": 173864, "Bacon, cooked": 167914, "Pork sausage": 174578,
    "Lamb, roasted": 174312,
    # Fish & seafood
    "Salmon, baked": 175168, "Tuna, canned in water": 171986, "Cod, baked": 171956,
    "Tilapia, baked": 175177, "Trout, baked": 173718, "Pollock, baked": 173681,
    "Mackerel, cooked": 175120, "Herring, pickled": 175118, "Sardines, canned in oil": 175139,
    "Shrimp, cooked": 175180,
    # Eggs & dairy
    "Egg, boiled": 173424, "Egg white": 172183, "Omelette": 172185, "Milk 2%": 171267,
    "Whole milk 3.2%": 171265, "Skim milk": 171269, "Kefir 2.5%": 170904,
    "Plain yogurt, whole milk": 171284, "Skyr": 170894, "Cottage cheese 4%": 172179,
    "Cottage cheese 1%": 173417, "Tvorog 5%": 172182, "Cheddar cheese": 173414,
    "Mozzarella, part-skim": 170847, "Feta cheese": 173420, "Parmesan cheese": 170848,
    "Cream cheese": 173418, "Sour cream 20%": 171257, "Butter": 173410, "Heavy cream": 170859,
    "Whey protein powder": 173177,
    # Legumes & soy
    "Lentils, boiled": 172421, "Chickpeas, boiled": 173757, "Black beans, boiled": 173735,
    "Kidney beans, boiled": 175194, "Green peas, boiled": 170420, "Tofu, firm": 172475,
    "Tempeh": 174272, "Edamame": 168411, "Hummus": 172454, "Peanut butter": 174294,
    # Vegetables
    "Broccoli, boiled": 169967, "Spinach, raw": 168462, "Carrot, raw": 170393, "Tomato": 170457,
    "Cucumber": 168409, "Red bell pepper": 170108, "Onion": 170000, "Garlic": 169230,
    "Cabbage": 169975, "Cauliflower": 169986, "Zucchini": 169291, "Eggplant": 169228,
    "Mushrooms": 169251, "Lettuce": 169249, "Beetroot, boiled": 169146, "Green beans": 169961,
    "Pumpkin": 168448, "Asparagus": 168389, "Avocado": 171705, "Sauerkraut": 169279,
    "Pickled cucumbers": 168558,
    # Fruit
    "Apple": 171688, "Orange": 169097, "Pear": 169118, "Strawberries": 167762,
    "Blueberries": 171711, "Raspberries": 167755, "Grapes": 174683, "Watermelon": 167765,
    "Mango": 169910, "Pineapple": 169124, "Kiwi": 168153, "Peach": 169928, "Cherries": 171719,
    "Plum": 169949, "Grapefruit": 174673, "Lemon": 167746, "Dates, dried": 171726,
    "Raisins": 168165, "Dried apricots": 173941,
    # Nuts & seeds
    "Almonds": 170567, "Walnuts": 170187, "Cashews": 170162, "Peanuts": 172430,
    "Hazelnuts": 170581, "Pistachios": 170184, "Sunflower seeds": 170562, "Pumpkin seeds": 170556,
    "Chia seeds": 170554, "Flaxseed": 169414,
    # Fats, sauces & sweets
    "Olive oil": 171413, "Sunflower oil": 171017, "Mayonnaise": 171009, "Ketchup": 168556,
    "Honey": 169640, "Sugar": 169655, "Jam": 169641, "Dark chocolate 70%": 170273,
    "Milk chocolate": 167587, "Vanilla ice cream": 167575, "Potato chips": 170649,
    "Popcorn, air-popped": 167959, "Protein bar": 173158,
    # Dishes
    "Cheese pizza": 170317, "Hamburger": 170694, "Chicken soup": 172909,
    # Drinks
    "Orange juice": 169098, "Apple juice": 173933, "Cola": 174852, "Beer": 168746,
    "Red wine": 173190, "Coffee, black": 171890, "Tea, unsweetened": 173227,
    "Almond milk, unsweetened": 174832, "Soy milk": 172446,
}


# Dishes recognised from photos → FNDDS record. Where FNDDS has no record of the dish itself,
# the closest one is used and its USDA description is given in the comment.
FNDDS_IDS: dict[str, int] = {
    "Apple pie": 2707995, "Pork ribs, cooked": 2705894, "Baklava": 2708044,
    "Beef carpaccio": 2706394,         # Steak tartare (both raw beef)
    "Steak tartare": 2706394,
    "Beet salad": 2709875,             # Beets, fresh, cooked, fat added
    "Beignets": 2708071, "Bibimbap": 2708950, "Bread pudding": 2705680,
    "Breakfast burrito": 2707343,      # Egg burrito
    "Bruschetta": 2707652,
    "Caesar salad": 2706818,           # Chicken or turkey caesar garden salad
    "Cannoli": 2708052,                # Pastry, Italian, with cheese
    "Caprese salad": 2705818,          # Mozzarella cheese, tomato, and basil, with oil and vinegar
    "Carrot cake": 2707859, "Ceviche": 2706463,
    "Cheese plate": 2705704,           # Cheese, NFS
    "Cheesecake": 2707861, "Chicken curry": 2706437, "Chicken quesadilla": 2708593,
    "Chocolate cake": 2707866, "Chocolate mousse": 2705698, "Churros": 2708070,
    "Clam chowder": 2707139, "Club sandwich": 2706993, "Crab cakes": 2706549,
    "Creme brulee": 2705684, "Ham and cheese sandwich": 2706965, "Cupcake": 2707853,
    "Deviled eggs": 2707181, "Doughnut": 2708062, "Dumplings, steamed": 2708708,
    "Eggs Benedict": 2707180, "Escargots": 2706368, "Falafel": 2707408,
    "Beef tenderloin steak": 2705841,
    "Fried fish": 2706244,             # Fish, cod, fried (fish and chips)
    "Liver pate": 2706155, "French onion soup": 2710108, "French toast": 2708335,
    "Fried calamari": 2706334, "Fried rice": 2708952, "Frozen yogurt": 2705452,
    "Garlic bread": 2707626, "Potato gnocchi": 2708722, "Greek salad": 2709830,
    "Grilled cheese sandwich": 2705801, "Guacamole": 2709307, "Gyoza, fried": 2708705,
    "Hot and sour soup": 2707136, "Hot dog": 2707056, "Huevos rancheros": 2707191,
    "Lasagna": 2708750, "Lobster bisque": 2707140,
    "Lobster roll": 2706825,           # Lobster salad
    "Macaroni and cheese": 2708811,
    "Macarons": 2707936,               # Cookie, macaroon
    "Miso soup": 2707455, "Mussels": 2706350, "Nachos": 2708576, "Onion rings": 2710055,
    "Oysters": 2706351, "Pad thai": 2708804, "Paella": 2706723,
    "Panna cotta": 2705682,            # Custard
    "Peking duck": 2706141, "Pho": 2707124,
    "Poutine": 2709469,                # Potato, french fries, with cheese
    "Roast beef": 2705847, "Pulled pork sandwich": 2706882, "Ramen": 2709153,
    "Ravioli with meat sauce": 2708764, "Red velvet cake": 2707884,
    "Risotto": 2708405,                # Rice, white, cooked, made with butter
    "Samosa": 2708730, "Tuna sashimi": 2706308, "Scallops, grilled": 2706357,
    "Seaweed salad": 2709990,          # Seaweed, cooked, fat added
    "Shrimp and grits": 2708369,       # Grits, with cheese, fat added
    "Spaghetti bolognese": 2708838,    # Pasta with tomato-based sauce and meat, home recipe
    "Spaghetti carbonara": 2708856,    # Pasta with cream sauce, home recipe
    "Spring rolls": 2708702,           # Egg roll, with beef and/or pork
    "Strawberry shortcake": 2707896, "Sushi": 2708959, "Tacos": 2708515,
    "Takoyaki": 2708024,               # Fritter, plain
    "Tiramisu": 2705702, "Tuna tartare": 2706308, "Waffles": 2708312,
    # Ukrainian dishes. "Soup, borscht" in FNDDS is the meatless beet soup (20 kcal); Ukrainian
    # borscht is cooked on meat broth, so the vegetable soup with meat is closer.
    "Ukrainian borscht": 2710116,      # Soup, vegetable, with meat
    "Potato pancakes (deruny)": 2709552,
    "Stuffed cabbage rolls (holubtsi)": 2706620,
    "Olivier salad": 2709532,          # Potato salad with egg, made with mayonnaise
    "Paska (Easter bread)": 2707624,   # Bread, egg, Challah
    "Shashlik": 2706779,               # Pork shish kabob with vegetables
    "Varenyky (pierogi)": 2708720,
}
MACRO_IDS = {"kcal": 1008, "proteins": 1003, "fats": 1004, "carbs": 1005}


def _round(value: float) -> float:
    if value >= 100:
        return round(value)
    if value >= 10:
        return round(value, 1)
    return round(value, 3 if value < 1 else 2)


def label_fields(name: str, values: dict[int, float]) -> tuple[dict[str, float], dict[str, str]]:
    """The four label fields of one food, and the sources of the estimated ones."""
    label = {field: values[nid] for field, nid in LABEL_IDS.items() if nid in values}
    if SODIUM in values:
        label["salt"] = values[SODIUM] / 1000 * SALT_PER_SODIUM
    sources = {}
    for field, (value, source) in LABEL_ESTIMATES.get(name, {}).items():
        if field in label:
            raise SystemExit(f"{name}: USDA reports {field}; drop its estimate")
        label[field], sources[field] = value, source
    missing = {*LABEL_IDS, "salt"} - label.keys()
    if missing:
        raise SystemExit(f"{name}: no USDA value for {sorted(missing)}; add it to LABEL_ESTIMATES")
    return {field: round(label[field], 2) for field in (*LABEL_IDS, "salt")}, sources


def read(data_dir: str, ids: dict[str, int], by_number: bool) -> tuple[dict, dict]:
    """Descriptions and nutrient amounts of the wanted records of one FoodData Central dataset.

    FNDDS files give the nutrient *number* (208, 203…) in food_nutrient.nutrient_id; it is turned
    into the nutrient id SR Legacy uses (1008, 1003…) through nutrient.csv."""
    wanted = {str(i) for i in ids.values()}
    number_to_id: dict[str, int] = {}
    if by_number:
        with open(os.path.join(data_dir, "nutrient.csv"), encoding="utf-8") as f:
            for row in csv.DictReader(f):
                number = row["nutrient_nbr"]
                if number and float(number).is_integer():  # 205 = carbs by difference, not 205.2
                    number_to_id[str(int(float(number)))] = int(row["id"])
    descriptions: dict[str, str] = {}
    with open(os.path.join(data_dir, "food.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["fdc_id"] in wanted:
                descriptions[row["fdc_id"]] = row["description"]
    amounts: dict[str, dict[int, float]] = {}
    with open(os.path.join(data_dir, "food_nutrient.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["fdc_id"] in wanted and row["amount"]:
                nid = number_to_id.get(row["nutrient_id"]) if by_number else int(row["nutrient_id"])
                if nid is not None:
                    amounts.setdefault(row["fdc_id"], {})[nid] = float(row["amount"])
    missing = wanted - descriptions.keys()
    if missing:
        raise SystemExit(f"FDC ids not found in {data_dir}: {sorted(missing)}")
    return descriptions, amounts


def check_dish_macros(amounts: dict[str, dict[int, float]]) -> None:
    """The dishes in seed_foods.GENERIC_FOODS carry the macros of their FNDDS record."""
    sys.path.insert(0, str(ROOT))
    from seed_foods import GENERIC_FOODS

    seeded = {name: dict(zip(MACRO_IDS, values)) for name, *values in GENERIC_FOODS}
    for name, fdc_id in FNDDS_IDS.items():
        if name not in seeded:
            raise SystemExit(f"{name}: not in seed_foods.GENERIC_FOODS")
        usda = amounts[str(fdc_id)]
        for field, nid in MACRO_IDS.items():
            if seeded[name][field] != round(usda[nid], 0 if field == "kcal" else 1):  # as written in seed_foods
                raise SystemExit(f"{name}: {field} is {seeded[name][field]} in seed_foods, {usda[nid]} in FNDDS")


def build(sr_dir: str, fndds_dir: str) -> tuple[dict, dict, dict]:
    """Micronutrients, label fields and estimate sources per food name."""
    descriptions, amounts = read(sr_dir, FDC_IDS, by_number=False)
    fndds_descriptions, fndds_amounts = read(fndds_dir, FNDDS_IDS, by_number=True)
    check_dish_macros(fndds_amounts)
    descriptions.update(fndds_descriptions)
    amounts.update(fndds_amounts)

    out, labels, estimated = {}, {}, {}
    for name, fdc_id in {**FDC_IDS, **FNDDS_IDS}.items():
        values = amounts.get(str(fdc_id), {})
        micros = {ours: values[nid] for ours, nid in NUTRIENT_IDS.items() if nid in values}
        if "Vitamin D" not in micros and VITAMIN_D_UG in values:
            micros["Vitamin D"] = values[VITAMIN_D_UG] * 40
        ala = values.get(ALA, values.get(ALA_18_3))
        omega = [v for v in (ala, values.get(EPA), values.get(DPA), values.get(DHA)) if v is not None]
        if omega:
            micros["Omega-3"] = sum(omega)
        out[name] = (fdc_id, descriptions[str(fdc_id)], {k: _round(v) for k, v in micros.items()})
        labels[name], sources = label_fields(name, values)
        if sources:
            estimated[name] = sources
    return out, labels, estimated


def write(data: dict, labels: dict, estimated: dict) -> None:
    lines = [
        '"""Micronutrients and label fields per 100 g for the built-in foods, from USDA FoodData Central',
        "(SR Legacy, 2018-04; dishes from FNDDS, 2024-10).",
        "",
        "Public domain (CC0). Generated by tools/build_usda_micros.py — do not edit by hand.",
        "USDA_MICROS maps our food name → (FDC id, USDA description, {nutrient: amount}); a nutrient USDA",
        "does not report for that food is absent, not zero.",
        "USDA_LABEL maps our food name → sugars, saturated fat, fibre and salt (g), all known: the few",
        "values USDA does not report are estimates, listed with their source in LABEL_ESTIMATED.",
        '"""',
        "",
        "USDA_MICROS: dict[str, tuple[int, str, dict[str, float]]] = {",
    ]
    for name, (fdc_id, desc, micros) in data.items():
        lines.append(f"    {name!r}: ({fdc_id}, {desc!r}, {micros!r}),")
    lines += ["}", "", "USDA_LABEL: dict[str, dict[str, float]] = {"]
    lines += [f"    {name!r}: {label!r}," for name, label in labels.items()]
    lines += ["}", "", "LABEL_ESTIMATED: dict[str, dict[str, str]] = {"]
    lines += [f"    {name!r}: {sources!r}," for name, sources in estimated.items()]
    lines.append("}")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    micros, labels, estimated = build(sys.argv[1], sys.argv[2])
    write(micros, labels, estimated)
    print(f"Wrote {len(micros)} foods to {OUTPUT.name}; "
          f"{sum(map(len, estimated.values()))} label values estimated for {len(estimated)} foods")
