#!/usr/bin/env python3
"""Convert the splits into the dataset directory LLaMA-Factory expects.

    data/generated/splits/{train,val}.jsonl
            |  this script
            v
    training/data/borakbot_{train,val}.json   sharegpt conversations
    training/data/dataset_info.json           the registration LLaMA-Factory reads

THE SYSTEM PROMPT IS STAMPED ONTO EVERY EXAMPLE, from the same
`_system_prompt.md` that `eval/generate.py` parses. This is the point of the
script. The two hallucination safeguards are supposed to be independent — the
weights teach refusal, the prompt states the boundary — but that only holds if
the model was trained with the prompt in context. Train without it and serve
with it, and every example the model saw is a distribution it will never meet
again; the prompt then reads as noise rather than as an instruction it has
learned to follow. One parsed copy, two consumers, no drift.

Test is deliberately NOT converted. It is scored by eval/, never seen by
LLaMA-Factory, and a `borakbot_test` entry sitting in dataset_info.json is an
invitation to accidentally list it under `dataset:` in the YAML.

    python training/to_llamafactory.py
    python training/to_llamafactory.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "data" / "generated" / "splits"
OUT_DIR = Path(__file__).resolve().parent / "data"

sys.path.insert(0, str(ROOT / "eval"))
from generate import load_system_prompt  # noqa: E402  single source of truth


def convert(rows: list[dict], system: str) -> list[dict]:
    return [
        {
            "conversations": [
                {"from": "human", "value": row["user"]},
                {"from": "gpt", "value": row["assistant"]},
            ],
            "system": system,
        }
        for row in rows
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    system = load_system_prompt()
    print(f"system prompt: {len(system)} chars, stamped onto every example")

    out_dir = Path(args.out_dir)
    written = {}
    for split in ("train", "val"):
        path = SPLITS_DIR / f"{split}.jsonl"
        if not path.exists():
            print(f"No split at {path}. Run scripts/split_corpus.py first.")
            return 2
        rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        converted = convert(rows, system)
        written[split] = converted
        refusal = sum(1 for r in rows if "source_probe" in r)
        print(f"  {split:<6} {len(converted):>4} examples  ({refusal} refusal)")

    info = {
        f"borakbot_{split}": {
            "file_name": f"borakbot_{split}.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }
        for split in written
    }

    if args.dry_run:
        print("\nFirst converted example:")
        print(json.dumps(written["train"][0], ensure_ascii=False, indent=2)[:600])
        print("\n--dry-run: nothing written.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, converted in written.items():
        path = out_dir / f"borakbot_{split}.json"
        path.write_text(json.dumps(converted, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")

    info_path = out_dir / "dataset_info.json"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {info_path.relative_to(ROOT)}")

    print("\nPoint LLaMA-Factory at this directory:")
    print(f"  dataset_dir: {out_dir.relative_to(ROOT).as_posix()}")
    print("  dataset: borakbot_train")
    print("  eval_dataset: borakbot_val")
    return 0


if __name__ == "__main__":
    sys.exit(main())
