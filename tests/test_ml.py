"""Web side of the Kolos ML models: Nutri-Score grades, diet quality and photo recognition."""
import io
from datetime import date, timedelta

import pytest

import food_db
import ml_inference
from app import diet_quality_7_days, product_payload
from conftest import create_user, login
from models import SOURCE_OFF, FoodLog, Product, db


@pytest.fixture(autouse=True)
def _no_models():
    """Every test starts (and ends) without trained models, like a fresh checkout."""
    ml_inference.register_score_model(None)
    ml_inference.register_photo_model(None)
    yield
    ml_inference.register_score_model(None)
    ml_inference.register_photo_model(None)


def _product(name, kcal=100, grade=None, source=None, **kw):
    p = Product(name=name, calories_per_100g=kcal, proteins=kw.get("p", 5), fats=kw.get("f", 5),
                carbs=kw.get("c", 5), nutri_grade=grade, nutri_source=source)
    db.session.add(p)
    db.session.flush()
    return p


def _log(uid, product, grams, day=None):
    db.session.add(FoodLog(user_id=uid, date=day or date.today(), meal_type="lunch",
                           product_id=product.id, portion_grams=grams))


def _jpeg():
    return io.BytesIO(b"\xff\xd8\xff\xe0fake-jpeg")


# --- grades ------------------------------------------------------------------------

def test_off_grade_is_parsed_and_stored(app):
    nutr = {"energy-kcal_100g": 60, "proteins_100g": 3, "fat_100g": 1, "carbohydrates_100g": 9}
    raw = {"code": "4820000000011", "product_name": "Kefir", "nutriments": nutr, "nutriscore_grade": "B"}
    assert food_db.parse_off_product(raw)["nutri_grade"] == "b"
    assert food_db.parse_off_product({**raw, "nutriscore_grade": "unknown"})["nutri_grade"] is None
    with app.app_context():
        product = food_db.upsert_off_product(food_db.parse_off_product(raw))
        db.session.commit()
        assert (product.source, product.nutri_grade, product.nutri_source) == (SOURCE_OFF, "b", "off")


def test_payload_uses_stored_grade_then_model(app):
    with app.test_request_context():
        rated = _product("Oats", grade="a", source="off")
        plain = _product("Cookie", kcal=480)
        assert (product_payload(rated)["grade"], product_payload(rated)["grade_src"]) == ("a", "off")
        assert product_payload(plain)["grade"] is None  # no model yet

        seen = []
        ml_inference.register_score_model(lambda features: seen.append(features) or "E")
        assert (product_payload(plain)["grade"], product_payload(plain)["grade_src"]) == ("e", "model")
        assert seen[0] == {"kcal": 480.0, "proteins": 5.0, "fats": 5.0, "carbs": 5.0,
                           "sugars": None, "saturated_fat": None, "fiber": None, "salt": None}
        assert product_payload(rated)["grade"] == "a"  # Open Food Facts wins over the model


def test_diet_quality_shares(app):
    with app.test_request_context():
        uid = create_user("q@user.test").id
        good = _product("Apple", kcal=50, grade="a", source="off")
        bad = _product("Chips", kcal=500, grade="e", source="off")
        unrated = _product("Soup", kcal=100)
        _log(uid, good, 200)                                     # 100 kcal A
        _log(uid, bad, 60, date.today() - timedelta(days=1))     # 300 kcal E
        _log(uid, unrated, 100)                                  # 100 kcal unrated
        _log(uid, bad, 100, date.today() - timedelta(days=9))    # outside the week
        db.session.commit()

        dq = diet_quality_7_days(uid)
        assert dq["total_kcal"] == 500
        assert dq["series"]["a"][-1] == 100 and dq["series"]["e"][-2] == 300
        assert dq["share"] == {"a": 20, "b": 0, "c": 0, "d": 0, "e": 60, "unrated": 20}
        assert (dq["rated_pct"], dq["good_pct"], dq["poor_pct"]) == (80, 25, 75)
        assert dq["model_ready"] is False


def test_dashboard_shows_diet_quality_instead_of_forecast(app, client):
    with app.app_context():
        uid = create_user("d@user.test").id
        _log(uid, _product("Apple", kcal=50, grade="a", source="off"), 200)
        db.session.commit()
    login(client, "d@user.test")
    html = client.get("/dashboard").get_data(as_text=True)
    assert "Diet quality" in html and 'id="qualityChart"' in html
    assert "Weight forecast" not in html and "weightChart" not in html
    assert 'class="grade-chip grade-a"' in html  # the logged food carries its grade


# --- nutrition label fields ---------------------------------------------------------

def test_off_label_fields_are_parsed_and_stored(app):
    nutr = {"energy-kcal_100g": 480, "proteins_100g": 6, "fat_100g": 22, "carbohydrates_100g": 62,
            "sugars_100g": 35, "saturated-fat_100g": 12.5, "sodium_100g": 0.2}
    raw = {"code": "4820000000028", "product_name": "Cookies", "nutriments": nutr}
    data = food_db.parse_off_product(raw)
    assert (data["sugars"], data["saturated_fat"], data["fiber"], data["salt"]) == (35, 12.5, None, 0.5)
    assert food_db.parse_off_product({**raw, "nutriments": {**nutr, "salt_100g": 0.4}})["salt"] == 0.4
    with app.app_context():
        product = food_db.upsert_off_product(data)
        db.session.commit()
        features = ml_inference.nutrition_features(product)
        assert (features["sugars"], features["saturated_fat"], features["fiber"], features["salt"]) == (35, 12.5, None, 0.5)


def test_custom_food_keeps_optional_label_fields(app, client):
    with app.app_context():
        create_user("c@user.test")
    login(client, "c@user.test")
    form = {"name": "Granola", "calories_per_100g": "450", "proteins": "10", "fats": "18", "carbs": "60"}
    client.post("/products/new", data={**form, "sugars": "20", "saturated_fat": "3", "fiber": "", "salt": "0.3"})
    client.post("/products/new", data={**form, "name": "Bad", "sugars": "70"})          # sugars > carbs
    client.post("/products/new", data={**form, "name": "Worse", "saturated_fat": "-1"})  # out of range
    with app.app_context():
        granola = Product.query.filter_by(name="Granola").one()
        assert (float(granola.sugars), float(granola.saturated_fat), granola.fiber, float(granola.salt)) == (20, 3, None, 0.3)
        assert Product.query.filter(Product.name.in_(["Bad", "Worse"])).count() == 0


# --- the exported Nutri-Score model --------------------------------------------------

def _export_tiny_model(directory, min_confidence=0.0):
    """A tiny model in the format of ml/nutriscore/export.py: grade E iff sugars ≥ 20 / kcal ≥ 400."""
    import gzip
    import json

    lgb = pytest.importorskip("lightgbm")
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    label = ["kcal", "proteins", "fats", "carbs", "sugars", "saturated_fat", "fiber", "salt"]
    specs = {"label": (label, 4, 20), "macros": (label[:4], 0, 400)}
    meta = {"format_version": 1, "grades": list("abcde"), "models": {}}
    for name, (features, column, cut) in specs.items():
        X = rng.uniform(0, 500 if column == 0 else 50, size=(400, len(features)))
        y = np.where(X[:, column] >= cut, 4, 0)
        booster = lgb.train({"objective": "multiclass", "num_class": 5, "verbose": -1, "min_data_in_leaf": 5},
                            lgb.Dataset(X, y), num_boost_round=20)
        with gzip.open(directory / f"{name}.txt.gz", "wt", encoding="utf-8") as fh:
            fh.write(booster.model_to_string())
        meta["models"][name] = {"file": f"{name}.txt.gz", "features": features,
                                "required": [f for f in features if f != "fiber"],
                                "min_confidence": min_confidence if name == "macros" else 0.0}
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_score_model_uses_label_model_then_confident_macros(app, tmp_path):
    import score_model

    _export_tiny_model(tmp_path)
    app.config["SCORE_MODEL_DIR"] = tmp_path
    assert score_model.load_score_model(app) and ml_inference.score_model_ready()
    with app.test_request_context():
        sweet = _product("Candy", kcal=380, p=1, f=1, c=90)
        sweet.sugars, sweet.saturated_fat, sweet.salt = 80, 0.5, 0.1   # fibre unknown: still the label model
        plain = _product("Bread", kcal=250, p=8, f=2, c=48)
        plain.sugars, plain.saturated_fat, plain.salt = 3, 0.4, 1.0
        rich = _product("Lard", kcal=900, p=0, f=99, c=0)                  # macros only
        assert ml_inference.product_grade(sweet) == ("e", "model")
        assert ml_inference.product_grade(plain) == ("a", "model")
        assert ml_inference.product_grade(rich) == ("e", "model")
        assert product_payload(rich)["grade_src"] == "model"


def test_unsure_macro_model_gives_no_grade(app, tmp_path):
    import score_model

    _export_tiny_model(tmp_path, min_confidence=1.01)  # nothing is ever confident enough
    app.config["SCORE_MODEL_DIR"] = tmp_path
    assert score_model.load_score_model(app)
    with app.test_request_context():
        assert ml_inference.product_grade(_product("Lard", kcal=900, p=0, f=99, c=0)) == (None, None)


def test_app_runs_without_exported_model(app, tmp_path):
    import score_model

    app.config["SCORE_MODEL_DIR"] = tmp_path / "missing"
    assert score_model.load_score_model(app) is False
    app.config["SCORE_MODEL_DIR"] = None
    assert score_model.load_score_model(app) is False
    assert not ml_inference.score_model_ready()


# --- photo recognition -------------------------------------------------------------

def test_photo_api_is_unavailable_without_a_model(app, client):
    with app.app_context():
        create_user("p@user.test")
    login(client, "p@user.test")
    res = client.post("/api/food/photo", data={"photo": (_jpeg(), "a.jpg", "image/jpeg")})
    assert res.status_code == 503 and res.get_json()["ready"] is False
    assert "Soon" in client.get("/log-food").get_data(as_text=True)


def test_photo_api_matches_catalogue_foods(app, client):
    with app.app_context():
        create_user("p@user.test")
        _product("Borscht", kcal=60, grade="b", source="off")
        db.session.commit()
    login(client, "p@user.test")
    calls = []
    ml_inference.register_photo_model(
        lambda image, k: calls.append((image, k)) or [("borscht", 0.81), ("dragon fruit stew", 0.12), ("x", 0.01), ("y", 0)])

    res = client.post("/api/food/photo", data={"photo": (_jpeg(), "a.jpg", "image/jpeg")})
    assert res.status_code == 200
    candidates = res.get_json()["candidates"]
    assert calls[0] == (b"\xff\xd8\xff\xe0fake-jpeg", 3)
    assert [c["label"] for c in candidates] == ["borscht", "dragon fruit stew", "x"]  # top 3
    assert candidates[0]["confidence"] == 0.81
    assert candidates[0]["product"]["name"] == "Borscht" and candidates[0]["product"]["grade"] == "b"
    assert candidates[1]["product"] is None


def test_photo_api_rejects_bad_uploads(app, client):
    with app.app_context():
        create_user("p@user.test")
    login(client, "p@user.test")
    ml_inference.register_photo_model(lambda image, k: [])
    assert client.post("/api/food/photo", data={}).status_code == 400
    res = client.post("/api/food/photo", data={"photo": (io.BytesIO(b"GIF89a"), "a.gif", "image/gif")})
    assert res.status_code == 400
    big = io.BytesIO(b"\xff" * (ml_inference.PHOTO_MAX_BYTES + 1))
    assert client.post("/api/food/photo", data={"photo": (big, "a.jpg", "image/jpeg")}).status_code == 413


def test_every_photo_class_is_a_builtin_food_with_usda_data():
    from app import PRODUCT_SEED
    from seed_foods import GENERIC_FOODS, PHOTO_FOODS, UK_NAMES
    from usda_micros import USDA_LABEL, USDA_MICROS

    builtin = {n for n, *_ in GENERIC_FOODS} | set(PRODUCT_SEED)
    assert len(PHOTO_FOODS) == 108  # 101 Food-101 classes + 7 Ukrainian dishes
    for cls, name in PHOTO_FOODS.items():
        assert name in builtin and name in USDA_MICROS and name in UK_NAMES, cls
        assert set(USDA_LABEL[name]) == {"sugars", "saturated_fat", "fiber", "salt"}, cls
    ukrainian = {"borscht", "varenyky", "holubtsi", "deruny", "olivier_salad", "paska", "shashlik"}
    assert ukrainian <= set(PHOTO_FOODS)
    assert UK_NAMES[PHOTO_FOODS["varenyky"]] == "Вареники"


def test_photo_api_offers_the_mapped_builtin_food(app, client):
    from app import _seed_generic_foods

    with app.app_context():
        create_user("p@user.test")
        _seed_generic_foods()
    login(client, "p@user.test")

    class Double:
        confident = 0.5

        def __call__(self, image, k):
            return [("varenyky", 0.67), ("ravioli", 0.21), ("gnocchi", 0.07)]

    ml_inference.register_photo_model(Double())
    res = client.post("/api/food/photo", data={"photo": (_jpeg(), "a.jpg", "image/jpeg")})
    body = res.get_json()
    assert res.status_code == 200 and body["likely"] is True
    names = [c["product"]["name"] for c in body["candidates"]]
    assert names == ["Varenyky (pierogi)", "Ravioli with meat sauce", "Potato gnocchi"]
    assert body["candidates"][0]["product"]["kcal"] == 195  # FNDDS "Pierogi"

    ml_inference.register_photo_model(lambda image, k: [("varenyky", 0.4), ("ravioli", 0.3), ("gnocchi", 0.1)])
    assert client.post("/api/food/photo", data={"photo": (_jpeg(), "a.jpg", "image/jpeg")}).get_json()["likely"] is False


def test_photo_api_rejects_a_file_that_is_not_an_image(app, client):
    with app.app_context():
        create_user("p@user.test")
    login(client, "p@user.test")

    def unreadable(image, k):
        raise ValueError("cannot read the photo")

    ml_inference.register_photo_model(unreadable)
    res = client.post("/api/food/photo", data={"photo": (_jpeg(), "a.jpg", "image/jpeg")})
    assert res.status_code == 400 and "error" in res.get_json()


def _tiny_onnx_model(directory, classes):
    """A stand-in for the exported network: mean colour of the 224 × 224 crop → softmax over 3 classes."""
    import json

    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("GlobalAveragePool", ["image"], ["pooled"]),
         helper.make_node("Flatten", ["pooled"], ["flat"]),
         helper.make_node("Mul", ["flat", "scale"], ["scaled"]),
         helper.make_node("Softmax", ["scaled"], ["probabilities"], axis=1)],
        "tiny",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, 224, 224])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["batch", 3])],
        [helper.make_tensor("scale", TensorProto.FLOAT, [1], [10.0])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(directory / "model.onnx"))
    meta = {"file": "model.onnx", "classes": classes, "confident": 0.5,
            "input": {"resize": 256, "crop": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_exported_photo_model_is_loaded_and_reads_real_images(app, tmp_path):
    pytest.importorskip("onnxruntime")
    from PIL import Image

    import photo_model

    _tiny_onnx_model(tmp_path, ["red", "green", "blue"])
    app.config["PHOTO_MODEL_DIR"] = tmp_path
    assert photo_model.load_photo_model(app) and ml_inference.photo_model_ready()
    assert ml_inference.photo_confidence_threshold() == 0.5

    photo = io.BytesIO()
    Image.new("RGB", (640, 480), (220, 30, 30)).save(photo, "JPEG")
    hits = ml_inference.recognize_photo(photo.getvalue(), top_k=2)
    assert [h["label"] for h in hits][0] == "red" and len(hits) == 2
    assert 0.5 < hits[0]["confidence"] <= 1
    with pytest.raises(ValueError):
        ml_inference.recognize_photo(b"\xff\xd8\xff\xe0fake-jpeg")


def test_photo_model_is_skipped_when_not_exported(app, tmp_path):
    import photo_model

    app.config["PHOTO_MODEL_DIR"] = tmp_path  # empty folder
    assert not photo_model.load_photo_model(app) and not ml_inference.photo_model_ready()
