#!/usr/bin/env python3
"""Validate generated training pairs against the knowledge base.

Catches the failure modes that matter for a generated corpus:

  1. Malformed JSONL.
  2. An answer missing a `check` string. These are substring-matched by the
     factual-correctness scorer (proposal S7, >=85% target), so a pair that
     fails here would be scored wrong at eval time no matter how good it reads.
  3. A `source_fact` that does not exist in the KB. Catches typos and IDs that
     were renamed after generation.
  4. Duplicate user turns. Ten paraphrases that collapse to fewer distinct
     questions teach less than the count suggests.
  5. A fallback line that is not byte-identical to the one quoted in Part A.
  6. Markdown, emoji or over-long answers — register violations.

Exit code is non-zero if any pair fails, so this can gate a commit.

    python stage2_chatbot/corpus_tooling/validate_pairs.py
    python stage2_chatbot/corpus_tooling/validate_pairs.py --dir stage2_chatbot/corpus/raw
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import yaml

STAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Anchored to stage2_chatbot/ rather than the working directory, so these run
# from anywhere. They used to require being launched from the repository root.
KB_DIR = os.path.join(STAGE, "knowledge_base")
DEFAULT_PAIR_DIR = os.path.join(STAGE, "corpus", "raw")

FALLBACK = "Maaf, saya tak pasti pasal tu lah — boleh tanya benda lain tak?"

# Answers are chat turns, not documents. Anything with markdown structure or an
# emoji has drifted out of register.
MARKDOWN = re.compile(r"(^|\n)\s*(#{1,6}\s|[-*+]\s|\d+\.\s)|\*\*|__|```")
MAX_SENTENCES = 3


def load_kb():
    """Return {fact_id: [check strings]} for every non-underscore domain file."""
    facts = {}
    for path in sorted(glob.glob(os.path.join(KB_DIR, "*.yaml"))):
        if os.path.basename(path).startswith("_"):
            continue
        doc = yaml.safe_load(open(path, encoding="utf-8")) or {}
        for fact in doc.get("facts", []):
            facts[fact["id"]] = list(fact.get("check", []))
    return facts


def load_probes():
    path = os.path.join(KB_DIR, "_out_of_scope.yaml")
    doc = yaml.safe_load(open(path, encoding="utf-8")) or {}
    return {p["q"] for p in doc.get("probes", [])}


def has_emoji(text: str) -> bool:
    return any(unicodedata.category(ch) == "So" for ch in text)


# Domains and URLs contain dots that are not sentence boundaries. Strip them
# before counting, or every answer citing imigresen-online.imi.gov.my is
# reported as a five-sentence paragraph.
URLISH = re.compile(r"\b(?:[\w-]+\.)+(?:my|com|gov|org|net)\b(?:/\S*)?")


def count_sentences(text: str) -> int:
    text = URLISH.sub("URL", text)
    return len([s for s in re.split(r"[.!?]+", text) if s.strip()])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_PAIR_DIR)
    args = ap.parse_args()

    facts = load_kb()
    probes = load_probes()
    if not facts:
        print(f"No facts found under {KB_DIR}/ — wrong working directory?")
        return 2

    errors: list[str] = []
    warnings: list[str] = []
    seen_users: dict[str, str] = {}
    per_fact = Counter()
    per_category = Counter()
    total = 0

    paths = sorted(glob.glob(os.path.join(args.dir, "*.jsonl")))
    if not paths:
        print(f"No .jsonl files under {args.dir}/")
        return 2

    for path in paths:
        name = os.path.basename(path)
        for lineno, raw in enumerate(open(path, encoding="utf-8"), 1):
            raw = raw.strip()
            if not raw:
                continue
            total += 1
            where = f"{name}:{lineno}"

            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"{where}  invalid JSON: {exc}")
                continue

            user = row.get("user", "")
            answer = row.get("assistant", "")

            if not user or not answer:
                errors.append(f"{where}  empty user or assistant")
                continue

            # Duplicate questions across the whole corpus.
            if user in seen_users:
                errors.append(f"{where}  duplicate user turn, first seen {seen_users[user]}")
            else:
                seen_users[user] = where

            if MARKDOWN.search(answer):
                errors.append(f"{where}  markdown in answer")
            if has_emoji(answer):
                errors.append(f"{where}  emoji in answer")
            if count_sentences(answer) > MAX_SENTENCES:
                warnings.append(f"{where}  {count_sentences(answer)} sentences (max {MAX_SENTENCES})")

            if "source_fact" in row:
                fid = row["source_fact"]
                per_fact[fid] += 1
                if fid not in facts:
                    errors.append(f"{where}  unknown source_fact {fid!r}")
                    continue
                for needle in facts[fid]:
                    if needle not in answer:
                        errors.append(
                            f"{where}  missing check string {needle!r}\n"
                            f"            fact={fid}\n"
                            f"            answer={answer}"
                        )
                if answer == FALLBACK:
                    errors.append(f"{where}  answered pair contains the fallback line")

            elif "source_probe" in row:
                per_category[row.get("category", "?")] += 1
                if row["source_probe"] not in probes:
                    warnings.append(f"{where}  source_probe not in _out_of_scope.yaml")
                if answer != FALLBACK:
                    errors.append(
                        f"{where}  fallback line not byte-identical\n"
                        f"            got      {answer!r}\n"
                        f"            expected {FALLBACK!r}"
                    )
            else:
                errors.append(f"{where}  row has neither source_fact nor source_probe")

    # --- report -------------------------------------------------------------
    answered = sum(per_fact.values())
    refusals = sum(per_category.values())

    print(f"Files          {len(paths)}")
    print(f"Pairs          {total}  ({answered} answered, {refusals} refusal)")
    if total:
        print(f"Refusal share  {refusals / total:.1%}")
    print(f"Facts covered  {len(per_fact)} of {len(facts)} in the KB")

    uncovered = sorted(set(facts) - set(per_fact))
    if uncovered:
        print(f"\nFacts with no pairs ({len(uncovered)}):")
        for fid in uncovered:
            print(f"  {fid}")

    thin = sorted(f for f, n in per_fact.items() if n < 10)
    if thin:
        print(f"\nFacts with fewer than 10 pairs ({len(thin)}):")
        for fid in thin:
            print(f"  {fid}  {per_fact[fid]}")

    if per_category:
        print("\nRefusal pairs by category:")
        for cat, n in sorted(per_category.items()):
            print(f"  {cat:<22} {n}")

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  {w}")

    if errors:
        print(f"\n{len(errors)} ERROR(S):")
        for e in errors:
            print(f"  {e}")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
