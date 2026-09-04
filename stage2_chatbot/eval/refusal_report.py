#!/usr/bin/env python3
"""Fallback accuracy and over-refusal, counted off the generations.

    eval/results/<tag>.json
            |  this script
            v
    eval/results/refusal_report.csv

The objective half of the bake-off, and the hallucination-control evidence at
Step 4. Two rates, and they trade against each other:

  fallback accuracy   of the out-of-scope probes, how many were declined?
                      Low means the boundary leaks and the model invents.
  over-refusal        of the in-domain questions, how many were declined?
                      High means the boundary is drawn too wide. This costs
                      helpfulness marks exactly as hallucination costs accuracy
                      marks, so reporting one without the other is not a result.

Refusal is counted two ways. EXACT is byte-equality with the canonical line,
which is what the proposal quotes and what the demo will be checked against;
Handoff 3 records that an em dash swapped for a hyphen already broke this once.
LOOSE accepts any recognisable decline. The gap between them is itself the
finding: a model that always declines correctly but never in the exact words
needs a formatting fix, while one that does not decline at all needs training.

The canonical line is read from the corpus, not hardcoded. All 130 refusal
pairs carry it identically, so the corpus is its own source of truth.

    python eval/refusal_report.py --runs qwen,llama,mallam
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent          # stage2_chatbot/, not the repo root
RESULTS_DIR = HERE / "results"
CORPUS = STAGE / "corpus" / "pairs_v1.jsonl"
OUT = RESULTS_DIR / "refusal_report.csv"

# Any one of these marks a reply as a decline, however it is worded. Kept
# deliberately generous: the loose rate is meant to be an upper bound on "did it
# refuse at all", so a false positive here is safer than a false negative.
LOOSE_MARKERS = [
    "tak pasti", "tak tahu", "tidak pasti", "tidak tahu",
    "maaf", "sorry", "tanya benda lain", "i don't know", "i'm not sure",
]


def canonical_fallback(corpus: Path = CORPUS) -> str:
    lines = [json.loads(l) for l in corpus.open(encoding="utf-8") if l.strip()]
    golds = Counter(r["assistant"] for r in lines if "source_probe" in r)
    if len(golds) != 1:
        raise SystemExit(
            f"{corpus}: expected one refusal wording, found {len(golds)}. "
            "The corpus must agree with itself before this rate means anything."
        )
    return golds.most_common(1)[0][0]


def squash(text: str) -> str:
    return " ".join(text.split()).strip()


def is_exact(pred: str, fallback: str) -> bool:
    return squash(pred) == squash(fallback)


def is_loose(pred: str) -> bool:
    low = squash(pred).lower()
    return any(m in low for m in LOOSE_MARKERS)


def report_run(payload: dict, fallback: str) -> dict:
    records = payload["records"]
    refusal = [r for r in records if r["stratum"] == "refusal"]
    answered = [r for r in records if r["stratum"] != "refusal"]

    def rate(rows, fn):
        return (sum(1 for r in rows if fn(r["prediction"])) / len(rows)) if rows else None

    return {
        "tag": payload["meta"].get("tag", "?"),
        "model": payload["meta"].get("model", ""),
        "adapter": payload["meta"].get("adapter") or "",
        "system_prompt": payload["meta"].get("system_prompt", ""),
        "n_refusal": len(refusal),
        "n_answered": len(answered),
        "fallback_exact": rate(refusal, lambda p: is_exact(p, fallback)),
        "fallback_loose": rate(refusal, is_loose),
        "over_refusal_exact": rate(answered, lambda p: is_exact(p, fallback)),
        "over_refusal_loose": rate(answered, is_loose),
    }


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated result tags")
    ap.add_argument("--show-misses", action="store_true",
                    help="print the in-domain questions that were wrongly declined")
    args = ap.parse_args()

    fallback = canonical_fallback()
    print(f"canonical fallback: {fallback}\n")

    rows, payloads = [], {}
    for tag in [t.strip() for t in args.runs.split(",") if t.strip()]:
        path = RESULTS_DIR / f"{tag}.json"
        if not path.exists():
            print(f"missing {path} — run eval/generate.py --tag {tag} first")
            return 2
        payloads[tag] = json.loads(path.read_text(encoding="utf-8"))
        rows.append(report_run(payloads[tag], fallback))

    print("=" * 76)
    print("REFUSAL BEHAVIOUR")
    print("=" * 76)
    print(f"{'tag':<14}{'refusals':>9}{'exact':>8}{'loose':>8}"
          f"{'in-dom':>9}{'over(x)':>9}{'over(l)':>9}")
    for r in rows:
        print(f"{r['tag']:<14}{r['n_refusal']:>9}{fmt(r['fallback_exact']):>8}"
              f"{fmt(r['fallback_loose']):>8}{r['n_answered']:>9}"
              f"{fmt(r['over_refusal_exact']):>9}{fmt(r['over_refusal_loose']):>9}")

    print("\nexact/loose = share of out-of-scope probes declined.")
    print("over(x)/over(l) = share of IN-DOMAIN questions wrongly declined; lower is better.")

    if args.show_misses:
        for tag, payload in payloads.items():
            wrong = [r for r in payload["records"]
                     if r["stratum"] != "refusal" and is_loose(r["prediction"])]
            if wrong:
                print(f"\n{tag}: {len(wrong)} in-domain question(s) declined")
                for r in wrong:
                    print(f"  [{r['stratum']}] {r['user']}")
                    print(f"      -> {squash(r['prediction'])[:110]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in r.items()})
    print(f"\nwrote {OUT.relative_to(STAGE.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
