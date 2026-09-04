#!/usr/bin/env python3
"""Assemble the reviewed training corpus from raw generation batches.

    stage2_chatbot/corpus/raw/*.jsonl   (gitignored, raw model output)
            |  this script
            v
    stage2_chatbot/corpus/pairs_v1.jsonl  (committed, what QLoRA trains on)

Raw output is not the training set. Two padding artefacts survive the pair
validator and have to be removed here, because both are invisible to an
exact-match duplicate check:

  1. CROSS-FACT TEMPLATES. A generator that runs out of ideas reuses one
     question frame across many facts, swapping only the quoted term:
     'ok kat WhatsApp orang guna "X", meaning dia apa sebenarnya?' appeared
     under 16 different slang facts. Sixteen distinct strings, one question.

  2. NEAR-DUPLICATES WITHIN A FACT. A "terse" variant that is just the direct
     question with the punctuation stripped: "lesen P berapa?" against
     "lesen P brape". Two rows, one question.

Both are dropped, and both are counted in the manifest so the reduction is
reported rather than hidden. Provenance (`batch`) is stamped onto every row so
a pair in the corpus can still be traced to the run that produced it.

    python stage2_chatbot/corpus_tooling/build_corpus.py
    python stage2_chatbot/corpus_tooling/build_corpus.py --dry-run
"""

from __future__ import annotations

import argparse
import difflib
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import date

STAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Anchored to stage2_chatbot/ rather than the working directory, so these run
# from anywhere. They used to require being launched from the repository root.
RAW_DIR = os.path.join(STAGE, "corpus", "raw")
OUT_PATH = os.path.join(STAGE, "corpus", "pairs_v1.jsonl")
MANIFEST_PATH = os.path.join(STAGE, "corpus", "pairs_v1_manifest.json")

# Concatenations of other batch files. Including these would double-count every
# pair they contain.
EXCLUDE = {"all_training_pairs.jsonl"}

# A question frame reused across at least this many facts is a template, not a
# question. Three is deliberately lenient — the observed case hit sixteen.
TEMPLATE_MIN_FACTS = 3

# Similarity above which two questions for the SAME fact are the same question.
NEAR_DUP_RATIO = 0.80


def skeleton(text: str) -> str:
    """Question shape with the fact-specific term removed."""
    text = text.lower()
    text = re.sub(r'"[^"]*"', "X", text)          # quoted term -> placeholder
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def normalise(text: str) -> str:
    """Order- and punctuation-insensitive form, for near-duplicate detection."""
    text = re.sub(r"[^\w\s]", "", text.lower())
    return " ".join(sorted(set(text.split())))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--raw-dir", default=RAW_DIR)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    paths = sorted(
        p for p in glob.glob(os.path.join(args.raw_dir, "*.jsonl"))
        if os.path.basename(p) not in EXCLUDE
    )
    if not paths:
        print(f"No batch files under {args.raw_dir}/")
        return 2

    skipped = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(args.raw_dir, "*.jsonl"))
        if os.path.basename(p) in EXCLUDE
    )

    rows = []
    for path in paths:
        batch = os.path.splitext(os.path.basename(path))[0]
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["batch"] = batch
            rows.append(row)

    before = len(rows)

    # --- 1. cross-fact templates -------------------------------------------
    by_skeleton = defaultdict(set)
    for row in rows:
        key = row.get("source_fact") or row.get("source_probe")
        by_skeleton[skeleton(row["user"])].add(key)

    templates = {s for s, keys in by_skeleton.items() if len(keys) >= TEMPLATE_MIN_FACTS}
    dropped_template = [r for r in rows if skeleton(r["user"]) in templates]
    rows = [r for r in rows if skeleton(r["user"]) not in templates]

    # --- 2. near-duplicates within a fact ----------------------------------
    kept, dropped_neardup = [], []
    seen_by_fact = defaultdict(list)
    for row in rows:
        key = row.get("source_fact") or row.get("source_probe")
        norm = normalise(row["user"])
        if any(
            difflib.SequenceMatcher(None, norm, prev).ratio() >= NEAR_DUP_RATIO
            for prev in seen_by_fact[key]
        ):
            dropped_neardup.append(row)
            continue
        seen_by_fact[key].append(norm)
        kept.append(row)

    rows = kept

    # --- report -------------------------------------------------------------
    per_fact = Counter(r["source_fact"] for r in rows if "source_fact" in r)
    answered = sum(per_fact.values())
    refusals = len(rows) - answered

    print(f"Batch files read      {len(paths)}")
    if skipped:
        print(f"Skipped (concat)      {', '.join(skipped)}")
    print(f"Pairs in              {before}")
    print(f"  dropped, template   {len(dropped_template)}")
    print(f"  dropped, near-dup   {len(dropped_neardup)}")
    print(f"Pairs out             {len(rows)}  ({answered} answered, {refusals} refusal)")
    if rows:
        print(f"Refusal share         {refusals / len(rows):.1%}")

    if templates:
        print(f"\nTemplates removed ({len(templates)}):")
        for s in sorted(templates):
            n = len(by_skeleton[s])
            print(f'  used across {n} facts:  "{s}"')

    if dropped_neardup:
        print(f"\nNear-duplicates removed ({len(dropped_neardup)}):")
        for r in dropped_neardup:
            key = r.get("source_fact") or r.get("source_probe")
            print(f"  {key:<34} {r['user']}")

    thin = sorted((n, f) for f, n in per_fact.items() if n < 8)
    if thin:
        print(f"\nFacts with fewer than 8 pairs after cleanup ({len(thin)}):")
        for n, f in thin:
            print(f"  {f:<34} {n}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "built": date.today().isoformat(),
        "output": args.out,
        "pairs_total": len(rows),
        "pairs_answered": answered,
        "pairs_refusal": refusals,
        "facts_covered": len(per_fact),
        "batches": [os.path.splitext(os.path.basename(p))[0] for p in paths],
        "excluded_files": skipped,
        "dropped_cross_fact_template": len(dropped_template),
        "dropped_near_duplicate": len(dropped_neardup),
        "rules": {
            "template_min_facts": TEMPLATE_MIN_FACTS,
            "near_dup_ratio": NEAR_DUP_RATIO,
        },
        "pairs_per_fact": dict(sorted(per_fact.items())),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print(f"\nWrote {args.out}")
    print(f"Wrote {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
