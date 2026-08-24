#!/usr/bin/env python3
"""Similarity metrics for the fine-tuned vs un-fine-tuned comparison.

    eval/results/<tag>.json          (generations, from eval/generate.py)
            |  this script
            v
    eval/results/score_report.csv

The graded Step 5 comparison: perplexity, BLEU, ROUGE-L and BERTScore, base
against adapter, tokenizer held constant. Refusal behaviour is NOT scored here --
that is eval/refusal_report.py, and the reason is the first decision below.

    # scoring only, CPU, no model load (~2 min, first run downloads mBERT)
    python eval/score.py --runs base,tuned

    # add perplexity, which needs the weights (GPU; ~4 min per run)
    python eval/score.py --runs base,tuned --perplexity \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --adapter-for tuned=YongVay/borakbot-qlora-r1 --4bit

    pip install sacrebleu rouge-score bert-score

FOUR DECISIONS THAT CHANGE THE NUMBERS

1.  Refusal rows are excluded from BLEU/ROUGE/BERTScore.
    All 11 refusal golds in the test split are the same canonical sentence. A
    model that declines everything would score near-perfectly on 11 of 63 items
    -- and the un-fine-tuned base very nearly IS that model, declining 82% of
    in-domain questions. Including refusals would hand it ~17% of the corpus for
    free and make over-refusal look like accuracy. Refusal is measured properly,
    both ways, by eval/refusal_report.py; scoring it twice with a metric that
    rewards the failure mode would corrupt the headline number.

2.  BERTScore runs on bert-base-multilingual-cased, not the default.
    bert_score defaults to roberta-large, which is English-only, and the
    references are Malay-English code-switched. Absolute values here are NOT
    comparable to published English BERTScore figures -- mBERT sits lower and the
    rescale baseline does not apply. Only the base-vs-tuned delta is meaningful,
    which is all the comparison claims.

3.  ROUGE-L runs without stemming.
    rouge_score's stemmer is Porter, which is English. On Malay it adds noise --
    it will not relate "memandu" to "pandu", and it will happily chop the tail
    off a Malay word that happens to end in -s or -ing.

4.  Perplexity is scored on the answer tokens only.
    The prompt is masked to -100. Left unmasked, most of the token count is the
    ~1055-character system prompt, which is byte-identical across both runs and
    would dilute the difference toward zero. Unlike the three overlap metrics,
    perplexity is computed over the WHOLE split including refusals: it measures
    the distribution the model learned, and the fallback line is part of what it
    was trained to produce.

A NOTE ON n. The test split is 63 items, 52 of them answerable. Corpus BLEU on
52 short answers is noisy, so chrF is reported alongside it -- same references,
character n-grams, far more stable at this size. If BLEU and chrF disagree in
direction, chrF is the one to trust and the disagreement is worth a sentence in
the write-up.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = HERE / "results"
OUT = RESULTS_DIR / "score_report.csv"

sys.path.insert(0, str(HERE))

# Shared with the generator on purpose. The perplexity path must format prompts
# exactly the way generation did, and a second copy would drift.
import generate  # noqa: E402

BERT_MODEL = "bert-base-multilingual-cased"


def answerable(records: list[dict]) -> list[dict]:
    """Everything that is not an out-of-scope probe. See decision 1."""
    return [r for r in records if r["stratum"] != "refusal"]


def overlap_metrics(preds: list[str], golds: list[str]) -> dict:
    import sacrebleu
    from rouge_score import rouge_scorer

    bleu = sacrebleu.corpus_bleu(preds, [golds])
    chrf = sacrebleu.corpus_chrf(preds, [golds])

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rl = [scorer.score(g, p)["rougeL"].fmeasure for g, p in zip(golds, preds)]

    return {
        "bleu": bleu.score,
        "chrf": chrf.score,
        "rougeL": 100 * sum(rl) / len(rl),
    }


def bertscore_f1(preds: list[str], golds: list[str]) -> float:
    from bert_score import score as _score

    _, _, f1 = _score(preds, golds, model_type=BERT_MODEL, verbose=False)
    return 100 * f1.mean().item()


def perplexity(model, tok, system: str | None, records: list[dict],
               chat_format: str = "auto") -> float:
    """Teacher-forced perplexity of the gold answers. See decision 4."""
    import torch

    total_nll, total_tokens = 0.0, 0
    for i, rec in enumerate(records, 1):
        prompt, _ = generate.build_inputs(tok, system, rec["user"], chat_format)

        # add_special_tokens=False: apply_chat_template already emits
        # <|begin_of_text|>, and letting the tokenizer add a second BOS shifts
        # every position off what training saw.
        p_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        f_ids = tok(prompt + rec["gold"] + (tok.eos_token or ""),
                    return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)

        labels = f_ids.clone()
        labels[:, : p_ids.shape[1]] = -100

        with torch.no_grad():
            loss = model(f_ids, labels=labels).loss

        # The denominator HF actually averaged over: labels shift left by one,
        # so the first position predicts nothing.
        n = int((labels[:, 1:] != -100).sum())
        if n == 0:
            continue
        total_nll += loss.item() * n
        total_tokens += n

        if i % 20 == 0 or i == len(records):
            print(f"  ppl {i}/{len(records)}", flush=True)

    return math.exp(total_nll / total_tokens)


def score_run(payload: dict, with_bertscore: bool = True) -> dict:
    records = answerable(payload["records"])
    preds = [r["prediction"] for r in records]
    golds = [r["gold"] for r in records]

    row = {
        "tag": payload["meta"].get("tag", "?"),
        "adapter": payload["meta"].get("adapter") or "(none)",
        "n_scored": len(records),
        "n_excluded_refusal": len(payload["records"]) - len(records),
    }
    row.update(overlap_metrics(preds, golds))
    row["bertscore_f1"] = bertscore_f1(preds, golds) if with_bertscore else None
    row["perplexity"] = None
    return row


def by_stratum(payload: dict) -> list[dict]:
    records = answerable(payload["records"])
    out = []
    for s in sorted({r["stratum"] for r in records}):
        rows = [r for r in records if r["stratum"] == s]
        m = overlap_metrics([r["prediction"] for r in rows], [r["gold"] for r in rows])
        out.append({"stratum": s, "n": len(rows), **m})
    return out


def fmt(v) -> str:
    return f"{'n/a':>9}" if v is None else f"{v:9.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated result tags")
    ap.add_argument("--by-stratum", action="store_true",
                    help="also break BLEU/chrF/ROUGE-L down by domain")
    ap.add_argument("--no-bertscore", action="store_true",
                    help="skip BERTScore (avoids a ~700 MB mBERT download)")
    ap.add_argument("--perplexity", action="store_true",
                    help="also compute perplexity; needs --model and the weights")
    ap.add_argument("--model", default=None, help="base model id, for --perplexity")
    ap.add_argument("--adapter-for", action="append", default=[], metavar="TAG=REPO",
                    help="attach this adapter when scoring TAG; repeatable")
    ap.add_argument("--4bit", dest="four_bit", action="store_true",
                    help="load in 4-bit for --perplexity, as generation did")
    args = ap.parse_args()

    tags = [t.strip() for t in args.runs.split(",") if t.strip()]
    adapters = dict(a.split("=", 1) for a in args.adapter_for)

    payloads = {}
    for tag in tags:
        path = RESULTS_DIR / f"{tag}.json"
        if not path.exists():
            print(f"missing {path} -- run eval/generate.py --tag {tag} first")
            return 2
        payloads[tag] = json.loads(path.read_text(encoding="utf-8"))

    # Same questions, same order, or the two columns are not a comparison.
    reference = [r["user"] for r in payloads[tags[0]]["records"]]
    for tag in tags[1:]:
        if [r["user"] for r in payloads[tag]["records"]] != reference:
            print(f"{tag}.json was generated over a different probe set than "
                  f"{tags[0]}.json -- regenerate both with the same --split/--all")
            return 2

    rows = []
    for tag in tags:
        print(f"scoring {tag} ...", flush=True)
        rows.append(score_run(payloads[tag], with_bertscore=not args.no_bertscore))

    if args.perplexity:
        if not args.model:
            print("--perplexity needs --model")
            return 2
        system = (generate.load_system_prompt()
                  if payloads[tags[0]]["meta"]["system_prompt"] else None)
        for tag, row in zip(tags, rows):
            print(f"\nperplexity for {tag} ...", flush=True)
            tok, model = generate.load_model(args.model, adapters.get(tag), args.four_bit)
            # Whole split here, refusals included -- see decision 4.
            row["perplexity"] = perplexity(
                model, tok, system, payloads[tag]["records"],
                payloads[tag]["meta"].get("chat_format_requested", "auto"))
            del model

    print("\n" + "=" * 72)
    print(f"SIMILARITY TO GOLD  (n={rows[0]['n_scored']} answerable; "
          f"{rows[0]['n_excluded_refusal']} refusal rows excluded)")
    print("=" * 72)
    print(f"{'tag':<14}{'BLEU':>9}{'chrF':>9}{'ROUGE-L':>9}{'BERTSc':>9}{'PPL':>9}")
    for r in rows:
        print(f"{r['tag']:<14}{fmt(r['bleu'])}{fmt(r['chrf'])}{fmt(r['rougeL'])}"
              f"{fmt(r['bertscore_f1'])}{fmt(r['perplexity'])}")
    print("\nHigher is better except PPL, where lower is better.")
    print(f"BERTScore F1 is {BERT_MODEL}: deltas are meaningful, absolutes are not.")

    if args.by_stratum:
        for tag in tags:
            print(f"\n{tag} by domain")
            print(f"  {'stratum':<24}{'n':>4}{'BLEU':>8}{'chrF':>8}{'ROUGE-L':>9}")
            for s in by_stratum(payloads[tag]):
                print(f"  {s['stratum']:<24}{s['n']:>4}{s['bleu']:>8.2f}"
                      f"{s['chrf']:>8.2f}{s['rougeL']:>9.2f}")

    # A partial run must not overwrite a complete report. --no-bertscore and a
    # missing --perplexity both leave None in a column, and writing that on top
    # of a finished 35-minute GPU run silently destroys numbers that cost real
    # time to produce. Exploratory runs (--by-stratum, a quick BLEU check) print
    # and stop; only the full set of metrics earns the file.
    missing = [name for name, key in (("BERTScore", "bertscore_f1"),
                                      ("perplexity", "perplexity"))
               if any(r[key] is None for r in rows)]
    if missing:
        print(f"\n{OUT.relative_to(ROOT)} left unchanged: this run computed no "
              f"{' and no '.join(missing)}, and a partial report would overwrite "
              "a complete one.")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in r.items()})
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
