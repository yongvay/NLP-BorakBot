#!/usr/bin/env python3
"""Split the reviewed corpus into train / val / test.

    data/generated/pairs_v1.jsonl
            |  this script
            v
    data/generated/splits/{train,val,test}.jsonl  (committed)

PAIR-LEVEL, NOT FACT-LEVEL. The split holds out some *paraphrases* of every
fact, not whole facts. This is deliberate and it does leak knowledge across the
split — see docs/DESIGN.md. In one line: knowledge is baked into the weights at
fine-tuning time with no retrieval step, so a held-out fact is one the model was
never told and provably cannot answer. That test is guaranteed to fail and
measures nothing. Held-out paraphrases measure whether the fact was learned
robustly across phrasings rather than memorised in one surface form, which is
the thing this architecture can actually be good at. Report it as paraphrase
robustness, never as generalisation to unseen knowledge.

Stratified by fact, so every fact appears in train and the answered/refusal
balance survives into each split. Small groups are the reason the quota uses a
carried remainder rather than per-group rounding: a fact with 5 pairs has a 10%
share of 0.5, and rounding that per group either starves the eval splits or
inflates them to 20%. Carrying the fraction across groups lands the global
totals on 80/10/10 while leaving every group at least 3 pairs in train.

    python scripts/split_corpus.py
    python scripts/split_corpus.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import date

IN_PATH = "data/generated/pairs_v1.jsonl"
OUT_DIR = "data/generated/splits"
MANIFEST_PATH = "data/generated/splits/manifest.json"

VAL_FRACTION = 0.10
TEST_FRACTION = 0.10

# Fixed so the split is reproducible from the committed corpus alone. Changing
# it invalidates every evaluation number already reported.
SEED = 20260822


def group_key(row: dict) -> str:
    """Answered pairs carry `source_fact`, refusal pairs carry `source_probe`."""
    return row.get("source_fact") or row.get("source_probe")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--in", dest="in_path", default=IN_PATH)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if not os.path.exists(args.in_path):
        print(f"No corpus at {args.in_path}")
        return 2

    rows = [
        json.loads(line)
        for line in open(args.in_path, encoding="utf-8")
        if line.strip()
    ]

    groups = defaultdict(list)
    for row in rows:
        key = group_key(row)
        if key is None:
            print(f"Row has neither source_fact nor source_probe: {row['user']!r}")
            return 1
        groups[key].append(row)

    rng = random.Random(args.seed)

    # Shuffle group order too. The remainder carry walks the groups in sequence,
    # so a fixed order correlated with domain would hand every leftover eval slot
    # to the same domain.
    keys = sorted(groups)
    rng.shuffle(keys)

    splits = {"train": [], "val": [], "test": []}
    carry_val = carry_test = 0.0

    for key in keys:
        items = groups[key][:]
        rng.shuffle(items)

        carry_val += VAL_FRACTION * len(items)
        n_val = int(carry_val)
        carry_val -= n_val

        carry_test += TEST_FRACTION * len(items)
        n_test = int(carry_test)
        carry_test -= n_test

        splits["val"].extend(items[:n_val])
        splits["test"].extend(items[n_val:n_val + n_test])
        splits["train"].extend(items[n_val + n_test:])

    # --- invariants ---------------------------------------------------------
    train_facts = {group_key(r) for r in splits["train"]}
    missing = sorted(set(groups) - train_facts)
    if missing:
        print(f"{len(missing)} fact(s) absent from train — the split is unusable:")
        for key in missing:
            print(f"  {key}")
        return 1

    assert sum(len(v) for v in splits.values()) == len(rows)

    # --- report -------------------------------------------------------------
    print(f"Corpus                {len(rows)} pairs, {len(groups)} facts/probes")
    print(f"Seed                  {args.seed}")
    print()
    print(f"{'split':<8}{'pairs':>7}{'share':>8}{'facts':>8}{'refusal':>10}")
    for name in ("train", "val", "test"):
        rows_ = splits[name]
        refusal = sum(1 for r in rows_ if "source_probe" in r)
        facts = len({group_key(r) for r in rows_})
        share = len(rows_) / len(rows)
        pct = refusal / len(rows_) if rows_ else 0.0
        print(f"{name:<8}{len(rows_):>7}{share:>7.1%}{facts:>8}{pct:>9.1%}")

    overall = sum(1 for r in rows if "source_probe" in r) / len(rows)
    print(f"\nRefusal share in the full corpus: {overall:.1%}")

    per_group = Counter(group_key(r) for r in splits["train"])
    thin = sorted((n, k) for k, n in per_group.items() if n < 4)
    if thin:
        print(f"\nFacts with fewer than 4 training pairs ({len(thin)}):")
        for n, key in thin:
            print(f"  {key:<34} {n}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    for name in ("train", "val", "test"):
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for row in splits[name]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {path}")

    manifest = {
        "built": date.today().isoformat(),
        "source": args.in_path,
        "seed": args.seed,
        "strategy": "pair-level, stratified by fact, carried-remainder quota",
        "fractions": {"train": round(1 - VAL_FRACTION - TEST_FRACTION, 2),
                      "val": VAL_FRACTION, "test": TEST_FRACTION},
        "counts": {name: len(splits[name]) for name in splits},
        "refusal_counts": {
            name: sum(1 for r in splits[name] if "source_probe" in r)
            for name in splits
        },
        "facts_total": len(groups),
        "facts_in_train": len(train_facts),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
