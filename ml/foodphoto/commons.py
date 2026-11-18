"""Step 1. Photos of Ukrainian dishes from Wikimedia Commons (free licences, attribution kept).

Food-101 has no Ukrainian dishes and there is no public dataset of them, so each dish is
collected from its Commons categories and their subcategories (CirrusSearch `deepcat:`, one
request per 50 files, licence data included). Rules:

* only JPEG/PNG/WebP photos at least MIN_SIDE px on the shorter side;
* a photo found under two dishes is dropped (ambiguous);
* every photo is then checked by eye on contact sheets (`--sheets`): the ones that show
  something else (a menu, a cooking process, a mostly empty plate…) go to review.json and are
  removed (`--apply-review`).

Author and licence of every kept photo go to reports/foodphoto/ua_sources.csv: CC BY / BY-SA
require attribution, and the report cites the list.

    python -m ml.foodphoto.commons --list          # counts per dish, nothing downloaded
    python -m ml.foodphoto.commons                 # download (resumes; skips existing files)
    python -m ml.foodphoto.commons --sheets        # contact sheets for the manual review
    python -m ml.foodphoto.commons --apply-review  # delete rejected photos, rewrite ua_sources.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import config

API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia User-Agent policy: name the tool and give a contact (here: the public repository).
USER_AGENT = "KolosDatasetBuilder/0.2 (https://github.com/dkrainii/course_project; diploma research)"
PAUSE = 1.0          # seconds between requests
MIN_SIDE = 224
THUMB_WIDTH = 640
REVIEW = Path(__file__).with_name("review.json")

# class name → Commons categories. English class names: the app looks them up in the catalogue.
DISHES: dict[str, list[str]] = {
    "borscht": ["Borscht", "Ukrainian borscht"],
    "varenyky": ["Varenyky", "Pierogi"],
    "holubtsi": ["Holubtsi", "Cabbage rolls"],
    "deruny": ["Deruny", "Draniki", "Potato pancakes"],
    "olivier_salad": ["Olivier salad"],
    "paska": ["Kulich", "Paska (bread)"],
    "shashlik": ["Shashlik"],
}
# Checked on 2026-11-18 and left out: fewer than 40 usable photos on Commons (syrnyky 30,
# nalysnyky 39, chicken Kyiv 25, pampushky 9, buckwheat porridge 12, kholodets 6, salo 2), or
# fewer than 40 after the review (pyrizhky 71 → ~34, pelmeni 185 → ~30: mostly shops, packaging
# and street festivals). A dish can come back with enough of our own photos in ua_own/.


# --------------------------------------------------------------------------- Commons API

def _get(url: str, params: dict | None = None) -> bytes:
    if params:
        url += "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(8):
        time.sleep(PAUSE)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(int(exc.headers.get("Retry-After") or 15) + 15 * (attempt + 1))
        except urllib.error.URLError:
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"Commons keeps refusing {url}")


def query(**params) -> dict:
    return json.loads(_get(API, {"action": "query", "format": "json", "maxlag": 5, **params}))


def category_files(category: str) -> dict[str, dict]:
    """Bitmap files in a category and all its subcategories, with their image info."""
    found, cont = {}, {}
    while True:
        data = query(generator="search", gsrsearch=f'deepcat:"{category}" filetype:bitmap', gsrnamespace=6,
                     gsrlimit=50, prop="imageinfo", iiprop="url|size|mime|extmetadata", iiurlwidth=THUMB_WIDTH,
                     iiextmetadatafilter="LicenseShortName|LicenseUrl|Artist|Credit", **cont)
        for warning in data.get("warnings", {}).values():
            print(f"[commons] {category}: {warning}", flush=True)
        for page in data.get("query", {}).get("pages", {}).values():
            if page.get("imageinfo"):
                found[page["title"]] = page["imageinfo"][0]
        if "continue" not in data:
            return found
        cont = data["continue"]


def dish_files() -> dict[str, dict[str, dict]]:
    found: dict[str, dict[str, dict]] = {}
    for dish, categories in DISHES.items():
        found[dish] = {}
        for category in categories:
            found[dish].update(category_files(category))
        print(f"[commons] {dish}: {len(found[dish])} files", flush=True)
    counts: dict[str, int] = {}
    for files in found.values():
        for title in files:
            counts[title] = counts.get(title, 0) + 1
    shared = {t for t, n in counts.items() if n > 1}
    if shared:
        print(f"[commons] {len(shared)} files belong to two dishes and are dropped", flush=True)
    return {dish: {t: ii for t, ii in files.items() if t not in shared} for dish, files in found.items()}


def plain(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


# --------------------------------------------------------------------------- steps

def file_name(title: str) -> str:
    stem = re.sub(r"[^\w.-]+", "_", Path(title.removeprefix("File:")).stem)[:100]
    return f"{stem}_{hashlib.md5(title.encode('utf-8')).hexdigest()[:6]}.jpg"  # x.jpg and x.png stay apart


def load_review() -> dict[str, str]:
    return json.loads(REVIEW.read_text(encoding="utf-8"))["rejected"] if REVIEW.exists() else {}


def download(list_only: bool = False) -> list[dict]:
    rejected = load_review()
    sources = []
    for dish, info in dish_files().items():
        usable = {t: ii for t, ii in sorted(info.items())
                  if t not in rejected and ii.get("mime") in ("image/jpeg", "image/png", "image/webp")
                  and min(ii.get("width", 0), ii.get("height", 0)) >= MIN_SIDE}
        if list_only:
            print(f"[commons] {dish}: {len(usable)} usable photos", flush=True)
            continue
        folder = config.UA_DIR / dish
        folder.mkdir(parents=True, exist_ok=True)
        kept = 0
        for title, ii in usable.items():
            meta = ii.get("extmetadata", {})
            target = folder / file_name(title)
            if not target.exists():
                target.write_bytes(_get(ii.get("thumburl") or ii["url"]))
            kept += 1
            sources.append({
                "dish": dish, "file": target.name, "title": title,
                "author": plain(meta.get("Artist", {}).get("value", "")) or plain(meta.get("Credit", {}).get("value", "")),
                "licence": plain(meta.get("LicenseShortName", {}).get("value", "")),
                "licence_url": meta.get("LicenseUrl", {}).get("value", ""),
                "url": ii.get("descriptionurl", ""),
            })
        print(f"[commons] {dish}: {kept} photos kept", flush=True)
    if not list_only:
        write_sources(sources)
    return sources


def write_sources(sources: list[dict]) -> None:
    config.SOURCES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(config.SOURCES_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["dish", "file", "title", "author", "licence", "licence_url", "url"])
        writer.writeheader()
        writer.writerows(sources)


def apply_review() -> None:
    rejected = load_review()
    with open(config.SOURCES_CSV, encoding="utf-8") as fh:
        sources = list(csv.DictReader(fh))
    kept = []
    for row in sources:
        # Rejected by eye, or a dish dropped from DISHES after the review (too few good photos).
        if row["title"] in rejected or row["dish"] not in DISHES:
            (config.UA_DIR / row["dish"] / row["file"]).unlink(missing_ok=True)
        else:
            kept.append(row)
    for folder in config.UA_DIR.iterdir():
        if folder.is_dir() and folder.name not in DISHES and not any(folder.iterdir()):
            folder.rmdir()
    write_sources(kept)
    print(f"[commons] review: {len(sources) - len(kept)} removed, {len(kept)} kept", flush=True)
    for dish in DISHES:
        print(f"[commons] {dish}: {sum(r['dish'] == dish for r in kept)}", flush=True)


def contact_sheets(columns: int = 10, rows_per_sheet: int = 8, cell: int = 160) -> None:
    """One numbered grid per dish (ml/data/foodphoto/review/<dish>_<page>.jpg) for checking by eye."""
    from PIL import Image, ImageDraw

    with open(config.SOURCES_CSV, encoding="utf-8") as fh:
        sources = list(csv.DictReader(fh))
    out = config.DATA_DIR / "review"
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for dish in DISHES:
        rows = [r for r in sources if r["dish"] == dish]
        for page in range(0, len(rows), columns * rows_per_sheet):
            chunk = rows[page:page + columns * rows_per_sheet]
            sheet = Image.new("RGB", (columns * cell, ((len(chunk) - 1) // columns + 1) * cell), "white")
            draw = ImageDraw.Draw(sheet)
            for i, row in enumerate(chunk):
                n = page + i
                with Image.open(config.UA_DIR / dish / row["file"]) as img:
                    img = img.convert("RGB")
                    img.thumbnail((cell - 4, cell - 4))
                    x, y = (i % columns) * cell, (i // columns) * cell
                    sheet.paste(img, (x + 2, y + 2))
                draw.rectangle((x + 2, y + 2, x + 34, y + 20), fill="black")
                draw.text((x + 5, y + 4), str(n), fill="white")
                index[f"{dish}/{n}"] = row["title"]
            sheet.save(out / f"{dish}_{page // (columns * rows_per_sheet):02d}.jpg", quality=85)
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[commons] contact sheets in {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true")
    group.add_argument("--sheets", action="store_true")
    group.add_argument("--apply-review", action="store_true")
    args = parser.parse_args()
    if args.sheets:
        contact_sheets()
    elif args.apply_review:
        apply_review()
    else:
        download(list_only=args.list)
