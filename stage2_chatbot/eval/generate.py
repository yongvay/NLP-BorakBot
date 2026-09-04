#!/usr/bin/env python3
"""Run a model over a fixed probe set and record what it says. No scoring.

    eval/probe_set.jsonl  (built once, committed, reused by every run)
            |  this script, once per model
            v
    eval/results/<tag>.json

Generation and scoring are separate scripts on purpose. Choosing a base model needs human judgement, not metrics: all the candidates are un-fine-tuned, so BLEU and BERTScore against rojak gold answers land in the same low band and mostly measure noise. Metrics only start to mean something once the same model is compared with and without the adapter, which is `eval/score.py` at Step 4. 

Splitting them also means the scorer is written after the base model is chosen, so it only ever has to handle one tokenizer.

The probe set is BUILT ONCE AND COMMITTED. If it were resampled per run, the
three models would be compared on different questions and the bake-off would
mean nothing. Delete eval/probe_set.jsonl only if you intend to invalidate
every result already recorded against it.

The system prompt is read out of stage2_chatbot/knowledge_base/_system_prompt.md rather
than copied here. It is also what gets stamped onto every training example, so
a second copy would eventually drift and train and inference would silently
disagree.

    # bake-off: same probe set, three candidates
    python eval/generate.py --model Qwen/Qwen2.5-3B-Instruct --tag qwen
    python eval/generate.py --model meta-llama/Llama-3.2-3B-Instruct --tag llama
    python eval/generate.py --model mesolitica/malaysian-... --tag mallam

    # step 4: full test split, base vs adapter, and the prompt ablation
    python eval/generate.py --model <winner> --split test --all --tag base
    python eval/generate.py --model <winner> --adapter <hf-repo> --split test --all --tag tuned
    python eval/generate.py --model <winner> --adapter <hf-repo> --split test --all \
        --no-system-prompt --refusals-only --tag tuned_noprompt
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# STAGE, not ROOT: this file sits under stage2_chatbot/, so parent.parent is the
# stage folder and not the repository root. Named for what it is.
STAGE = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT_MD = STAGE / "knowledge_base" / "_system_prompt.md"
SPLITS_DIR = STAGE / "corpus" / "splits"
PROBE_SET = Path(__file__).resolve().parent / "probe_set.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

PROBE_N = 20
SEED = 20260822

# Answers are capped at three sentences by the generation prompt, so this is
# generous. It exists to stop an un-fine-tuned base model rambling for a page --
# which several will, since nothing has taught them the length convention yet.
MAX_NEW_TOKENS = 160


def load_system_prompt(path: Path = SYSTEM_PROMPT_MD) -> str:
    """Extract the fenced prompt under '## Draft v1'.

    The file is prose with the prompt embedded in it, because its reasoning is
    examinable material. Parsing the block keeps one copy of the actual text.
    """
    text = path.read_text(encoding="utf-8")
    after = text.split("## Draft v1", 1)
    if len(after) != 2:
        raise SystemExit(f"{path}: no '## Draft v1' heading")
    m = re.search(r"```\s*\n(.*?)\n```", after[1], re.S)
    if not m:
        raise SystemExit(f"{path}: no fenced block under '## Draft v1'")
    return m.group(1).strip()


def stratum(row: dict) -> str:
    """Probe-set strata: one per factual domain, plus refusals as their own."""
    if "source_probe" in row:
        return "refusal"
    return row.get("domain", "unknown")


def build_probe_set(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Stratified sample, largest-remainder allocation so every stratum appears.

    Proportional allocation alone would round jpn_identity and
    immigration_passport (5 rows each in test) down to one or zero, and those
    two domains are exactly the thin ones whose behaviour needs watching.
    """
    by = defaultdict(list)
    for row in rows:
        by[stratum(row)].append(row)

    total = len(rows)
    quotas, remainders = {}, {}
    for key, items in by.items():
        exact = n * len(items) / total
        quotas[key] = max(1, int(exact))
        remainders[key] = exact - int(exact)

    # Hand out or claw back the difference by remainder size.
    order = sorted(by, key=lambda k: -remainders[k])
    while sum(quotas.values()) < n:
        for key in order:
            if sum(quotas.values()) >= n:
                break
            if quotas[key] < len(by[key]):
                quotas[key] += 1
    while sum(quotas.values()) > n:
        for key in reversed(order):
            if sum(quotas.values()) <= n:
                break
            if quotas[key] > 1:
                quotas[key] -= 1

    rng = random.Random(seed)
    out = []
    for key in sorted(by):
        items = by[key][:]
        rng.shuffle(items)
        out.extend(items[:quotas[key]])
    rng.shuffle(out)
    return out


def load_probe_set(split_path: Path, n: int, seed: int, use_all: bool) -> list[dict]:
    rows = [json.loads(l) for l in split_path.open(encoding="utf-8") if l.strip()]
    if use_all:
        return rows
    if PROBE_SET.exists():
        cached = [json.loads(l) for l in PROBE_SET.open(encoding="utf-8") if l.strip()]
        print(f"probe set: {len(cached)} items (reusing {PROBE_SET.name})")
        return cached
    probes = build_probe_set(rows, n, seed)
    with PROBE_SET.open("w", encoding="utf-8") as fh:
        for row in probes:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"probe set: built {len(probes)} items -> {PROBE_SET.name}  (commit this)")
    return probes


def load_model(model_id: str, adapter: str | None, four_bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    kwargs: dict = {}
    if four_bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
    kwargs["device_map"] = "auto" if torch.cuda.is_available() else None

    print(f"loading {model_id}{' (4-bit)' if four_bit else ''} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if adapter:
        from peft import PeftModel
        print(f"attaching adapter {adapter} ...", flush=True)
        model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, model


def build_inputs(tok, system: str | None, user: str, chat_format: str = "auto"):
    """Format one turn in the delimiters the checkpoint was trained on.

    Returns (text, format_used) so the run records which path it took.

    THE PLAIN FALLBACK IS NOT A NEUTRAL DEFAULT — it is a last resort, and the
    first bake-off proved it. `mallam-3b-20k-instructions` ships no
    `chat_template`, so it landed on the plain layout and produced degenerate
    output on all 20 probes: it never emitted a turn-ending token and instead
    carried on inventing 'User:' exchanges with itself. That measured the
    formatting fallback, not the model.

    Its model card states it uses the exact Mistral Instruct template, so pass
    `--chat-format mistral`. Mistral has no system role: the convention is to
    prepend the system text to the first user message, which is what happens
    here. If a checkpoint has no declared template, look up what it was trained
    on rather than accepting `plain`.
    """
    if chat_format == "mistral":
        content = f"{system}\n\n{user}" if system else user
        # No literal <s>: the tokenizer prepends BOS itself, and two of them
        # shift every position the model was trained to expect.
        return f"[INST] {content} [/INST]", "mistral"

    if chat_format == "auto" and getattr(tok, "chat_template", None):
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), "auto"

    parts = ([system] if system else []) + [f"User: {user}", "Assistant:"]
    return "\n\n".join(parts), "plain"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id of the base model")
    ap.add_argument("--adapter", default=None, help="HF repo or path of a LoRA adapter")
    ap.add_argument("--tag", required=True, help="short name; results/<tag>.json")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--all", action="store_true",
                    help="run the whole split, not the 20-item probe set")
    ap.add_argument("--refusals-only", action="store_true",
                    help="keep only refusal rows (for the system-prompt ablation)")
    ap.add_argument("--no-system-prompt", action="store_true",
                    help="ablation: generate without the constrained system prompt")
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--chat-format", default="auto", choices=["auto", "mistral", "plain"],
                    help="auto uses the tokenizer's own template; mistral forces the "
                         "[INST] layout for checkpoints that ship none (MaLLaM)")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true",
                    help="build the probe set and print prompts; load no model")
    args = ap.parse_args()

    split_path = SPLITS_DIR / f"{args.split}.jsonl"
    if not split_path.exists():
        print(f"No split at {split_path}. Run corpus_tooling/split_corpus.py first.")
        return 2

    rows = load_probe_set(split_path, PROBE_N, args.seed, args.all)
    if args.refusals_only:
        rows = [r for r in rows if "source_probe" in r]
        print(f"refusals only: {len(rows)} items")
    if not rows:
        print("Nothing to generate.")
        return 1

    system = None if args.no_system_prompt else load_system_prompt()

    if args.dry_run:
        print(f"\nsystem prompt: {'OFF' if system is None else f'{len(system)} chars'}")
        for row in rows[:3]:
            print(f"\n  [{stratum(row):<22}] {row['user']}")
            print(f"  gold: {row['assistant']}")
        print(f"\n--dry-run: {len(rows)} items, no model loaded.")
        return 0

    tok, model = load_model(args.model, args.adapter, args.four_bit)

    import torch
    records, templated = [], None
    for i, row in enumerate(rows, 1):
        text, used_template = build_inputs(tok, system, row["user"], args.chat_format)
        if templated is None:
            templated = used_template
            print(f"prompt format: {used_template}")
            if used_template == "plain":
                print("WARNING: this checkpoint declares no chat template and none was "
                      "forced. Expect degenerate output — the model has no turn-ending "
                      "token in this layout. Look up what it was trained on and pass "
                      "--chat-format.")
        enc = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                # Greedy. Sampling would make the bake-off unreproducible and
                # would confound model differences with seed differences.
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        reply = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        records.append({
            "stratum": stratum(row),
            "source": row.get("source_fact") or row.get("source_probe"),
            "category": row.get("category"),
            "user": row["user"],
            "gold": row["assistant"],
            "prediction": reply,
        })
        if i % 5 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.tag}.json"
    payload = {
        "meta": {
            "tag": args.tag,
            "model": args.model,
            "adapter": args.adapter,
            "split": args.split,
            "probe_set": "full-split" if args.all else PROBE_SET.name,
            "refusals_only": args.refusals_only,
            "system_prompt": not args.no_system_prompt,
            "chat_template": templated,
            "chat_format_requested": args.chat_format,
            "four_bit": args.four_bit,
            "decoding": "greedy",
            "max_new_tokens": args.max_new_tokens,
            "n": len(records),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "records": records,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path.relative_to(STAGE.parent)}  ({len(records)} generations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
