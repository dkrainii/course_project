"""Step 1. Open Food Facts CSV dump → compact Parquet with the graded products only.

The dump is ~1.5 GB gzipped (≈12 GB of text, 211 columns), so it is streamed in chunks and only
the columns from config.RAW_COLUMNS are kept. Rows without a Nutri-Score grade a–e are counted
(they are the products the model is meant for) but not stored.

    python -m ml.nutriscore.extract [path/to/dump.csv.gz]
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config

CHUNK_ROWS = 200_000
MACRO_COLUMNS = ["proteins_100g", "fat_100g", "carbohydrates_100g"]

SCHEMA = pa.schema(
    [(c, pa.string()) for c in config.TEXT_COLUMNS] + [(c, pa.float64()) for c in config.NUMERIC_COLUMNS]
)


def read_chunks(path: Path):
    return pd.read_csv(
        path, sep="\t", usecols=config.RAW_COLUMNS, dtype=str, chunksize=CHUNK_ROWS,
        quoting=csv.QUOTE_NONE,  # the export is not quoted; names contain stray quote characters
        encoding="utf-8", encoding_errors="replace", on_bad_lines="skip",
        na_values=[""], keep_default_na=False,
    )


def extract(raw_path: Path = config.RAW_DUMP, out_path: Path = config.EXTRACTED) -> dict:
    config.ensure_dirs()
    started = time.perf_counter()
    total = with_macros = with_macros_ungraded = kept = 0
    grade_values: Counter[str] = Counter()

    with pq.ParquetWriter(out_path, SCHEMA, compression="zstd") as writer:
        for chunk in read_chunks(raw_path):
            total += len(chunk)
            for col in config.NUMERIC_COLUMNS:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
            grade = chunk["nutriscore_grade"].str.strip().str.lower()
            grade_values.update(grade.fillna("(empty)").tolist())
            graded = grade.isin(config.GRADES)

            has_energy = chunk[["energy-kcal_100g", "energy_100g"]].notna().any(axis=1)
            has_macros = has_energy & chunk[MACRO_COLUMNS].notna().all(axis=1)
            with_macros += int(has_macros.sum())
            with_macros_ungraded += int((has_macros & ~graded).sum())

            part = chunk.loc[graded, config.RAW_COLUMNS].copy()
            part["nutriscore_grade"] = grade[graded]
            kept += len(part)
            writer.write_table(pa.Table.from_pandas(part, schema=SCHEMA, preserve_index=False))
            print(f"\r{total:>9,} rows read, {kept:>9,} graded", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    stat = raw_path.stat()
    stats = {
        "dump_file": raw_path.name,
        "dump_bytes": stat.st_size,
        "dump_modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "rows_total": total,
        "rows_graded": kept,
        "rows_with_macros": with_macros,
        "rows_with_macros_ungraded": with_macros_ungraded,
        "grade_values": dict(grade_values.most_common()),
        "seconds": round(time.perf_counter() - started, 1),
    }
    (config.RESULTS_DIR / "extract_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


if __name__ == "__main__":
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else config.RAW_DUMP
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else config.EXTRACTED
    print(json.dumps(extract(source, target), indent=2))
