#!/usr/bin/env python3
"""Aggregate the Likert sheets, un-blind, and report per-system means.

    eval/rating_sheet_<initials>.csv  (one per rater, independently filled)
            |  this script
            v
    eval/results/ratings_summary.csv

Reports inter-rater agreement alongside the means, and reports it FIRST. Two
raters who disagree by more than about a point are not measuring the same thing,
and a mean over two such raters is a number with nothing behind it. If agreement
is poor, the fix is to agree on what a 3 means and re-rate, not to average
harder and hope.

Agreement is given three ways because each hides a different failure:
  exact %        strict, but punishes a 4-vs-5 as hard as a 1-vs-5
  mean |diff|    magnitude, but a constant offset (one rater is just harsher)
                 looks identical to random noise
  Spearman rho   whether they RANK the systems the same way, which is the only
                 thing the bake-off actually needs them to agree on

    python eval/score_ratings.py
    python eval/score_ratings.py --sheets eval/rating_sheet_yv.csv,eval/rating_sheet_pq.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
KEY = HERE / "rating_key.json"
OUT = HERE / "results" / "ratings_summary.csv"

AXES = ["fluency", "rojak_naturalness", "factual_consistency"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", default=None,
                    help="comma-separated rating sheets (default: rating_sheet_*.csv)")
    ap.add_argument("--key", default=str(KEY))
    args = ap.parse_args()

    paths = ([Path(p.strip()) for p in args.sheets.split(",") if p.strip()]
             if args.sheets else
             sorted(Path(p) for p in glob.glob(str(HERE / "rating_sheet_*.csv"))))
    if not paths:
        print("No rating sheets found (eval/rating_sheet_<initials>.csv).")
        return 2

    frames = []
    for path in paths:
        df = pd.read_csv(path, encoding="utf-8-sig")
        rater = path.stem.replace("rating_sheet_", "") or path.stem
        missing = [a for a in AXES if a not in df.columns]
        if missing:
            print(f"{path.name}: missing column(s) {missing}")
            return 1
        blank = df[AXES].isna().any(axis=1).sum()
        if blank:
            print(f"{path.name}: {blank} row(s) not rated — fill them or drop them")
            return 1
        for axis in AXES:
            bad = df[~df[axis].between(1, 5)]
            if len(bad):
                print(f"{path.name}: {len(bad)} rating(s) outside 1-5 in '{axis}'")
                return 1
        df["rater"] = rater
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    raters = sorted(data["rater"].unique())
    print(f"raters: {', '.join(raters)}   rows each: {len(data) // len(raters)}\n")

    # ---- agreement, before any averaging --------------------------------
    if len(raters) == 2:
        a, b = (data[data.rater == r].set_index("row_id").sort_index() for r in raters)
        print("=" * 64)
        print(f"INTER-RATER AGREEMENT  ({raters[0]} vs {raters[1]})")
        print("=" * 64)
        print(f"{'axis':<22}{'exact':>9}{'mean |diff|':>14}{'spearman':>11}")
        for axis in AXES:
            x, y = a[axis].astype(float), b[axis].astype(float)
            exact = (x == y).mean()
            mad = (x - y).abs().mean()
            rho = x.corr(y, method="spearman")
            print(f"{axis:<22}{exact:>8.0%}{mad:>14.2f}{rho:>11.2f}")
        print("\nAgreement below ~0.4 rho, or mean |diff| above ~1.0, means the")
        print("scale was understood differently. Re-anchor and re-rate.\n")
    else:
        print(f"{len(raters)} rater(s) — agreement not computed.\n")

    # ---- un-blind and summarise -----------------------------------------
    key_path = Path(args.key)
    if key_path.exists():
        blind_map = json.loads(key_path.read_text(encoding="utf-8"))["blind_map"]
        reverse = {v: k for k, v in blind_map.items()}
        data["model"] = data["system"].map(reverse).fillna(data["system"])
    else:
        print(f"no {key_path.name} — leaving systems blinded")
        data["model"] = data["system"]

    summary = (data.groupby("model")[AXES]
               .agg(["mean", "std"])
               .round(2))
    summary.columns = [f"{a}_{s}" for a, s in summary.columns]
    summary["overall_mean"] = data.groupby("model")[AXES].mean().mean(axis=1).round(2)
    summary = summary.sort_values("overall_mean", ascending=False)

    print("=" * 64)
    print("LIKERT MEANS BY SYSTEM  (1-5, higher is better)")
    print("=" * 64)
    print(summary.to_string())

    by_stratum = (data.groupby(["model", "stratum"])[AXES].mean().round(2))
    print("\n" + "=" * 64)
    print("BY STRATUM")
    print("=" * 64)
    print(by_stratum.to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT)
    by_stratum.to_csv(OUT.with_name("ratings_by_stratum.csv"))
    print(f"\nwrote {OUT.name}, ratings_by_stratum.csv")

    best = summary.index[0]
    spread = summary["overall_mean"].iloc[0] - summary["overall_mean"].iloc[-1]
    print(f"\nHighest rated: {best}  (overall {summary['overall_mean'].iloc[0]})")
    if spread < 0.3:
        print(f"Spread is only {spread:.2f} — treat these as tied and choose on")
        print("tooling risk instead. Say that in docs/Archive/model_selection.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
