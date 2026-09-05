#!/usr/bin/env python3
"""Merge generation runs into one blinded Likert rating sheet.

    eval/results/*.json   (one per model, same probe set)
            |  this script
            v
    eval/rating_sheet.csv   blinded, shuffled, one row per (item, model)
    eval/rating_key.json    the un-blinding map

BLINDED AND SHUFFLED, because the raters are the two people who chose the
candidate models and wrote the corpus. Knowing which system produced a reply is
enough to move a subjective 1-5 rating, and "we rated them ourselves, unblinded"
is the first thing a marker will push on. Rating rows carry a system id like
`sys_c`; `rating_key.json` maps those back and is not needed until scoring.

Three axes, matching the proposal's human evaluation:
  fluency               is it well-formed, grammatical Malaysian speech?
  rojak_naturalness     would a Malaysian actually say it this way?
  factual_consistency   does it agree with the gold answer? For refusal items
                        this asks whether it declined rather than invented.

Rate 1-5. Both members rate every row independently, in separate copies of the
sheet, then run eval/score_ratings.py to get means and agreement.

    python eval/make_rating_sheet.py --runs qwen,llama,mallam
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SHEET = HERE / "rating_sheet.csv"
KEY = HERE / "rating_key.json"

SEED = 20260822

COLUMNS = [
    "row_id", "system", "stratum", "user", "gold", "prediction",
    "fluency", "rojak_naturalness", "factual_consistency", "notes",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="comma-separated result tags, e.g. qwen,llama,mallam")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(SHEET))
    args = ap.parse_args()

    tags = [t.strip() for t in args.runs.split(",") if t.strip()]
    runs = {}
    for tag in tags:
        path = RESULTS_DIR / f"{tag}.json"
        if not path.exists():
            print(f"missing {path} — run eval/generate.py --tag {tag} first")
            return 2
        runs[tag] = json.loads(path.read_text(encoding="utf-8"))

    # Every run must have answered the same questions, or the comparison is
    # between different tests rather than different models.
    keys = {tag: [r["user"] for r in run["records"]] for tag, run in runs.items()}
    first = keys[tags[0]]
    for tag in tags[1:]:
        if keys[tag] != first:
            print(f"probe sets differ: '{tags[0]}' and '{tag}' answered different items.")
            print("Re-run both against the committed eval/probe_set.jsonl.")
            return 1

    rng = random.Random(args.seed)
    blind = {tag: f"sys_{chr(ord('a') + i)}" for i, tag in enumerate(sorted(tags))}

    rows = []
    for tag, run in runs.items():
        for i, rec in enumerate(run["records"]):
            rows.append({
                "row_id": f"{i:02d}_{blind[tag]}",
                "system": blind[tag],
                "stratum": rec["stratum"],
                "user": rec["user"],
                "gold": rec["gold"],
                "prediction": rec["prediction"],
                "fluency": "", "rojak_naturalness": "",
                "factual_consistency": "", "notes": "",
            })
    rng.shuffle(rows)

    out = Path(args.out)
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    KEY.write_text(json.dumps({
        "seed": args.seed,
        "blind_map": blind,
        "items": len(first),
        "rows": len(rows),
    }, indent=2), encoding="utf-8")

    print(f"wrote {out.name}  ({len(rows)} rows = {len(first)} items x {len(tags)} systems)")
    print(f"wrote {KEY.name}  (do not open this before rating)")
    print("\nEach member takes their own copy, rates every row 1-5 on the three")
    print("axes, and saves it as rating_sheet_<initials>.csv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
