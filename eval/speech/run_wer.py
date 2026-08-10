"""Whisper evaluation for BorakBot -- WER across model/prompt configurations.

Produces the speech-evaluation tables required by proposal S7.4:
  results_detail.csv       one row per clip per config (transcripts + error counts)
  results_summary.csv      WER per config, strict and lenient
  results_by_category.csv  WER per config per switch_type  <- the interesting table
  substitutions.csv        every ref->hyp word swap, by frequency

Design notes worth defending at the demo:

* 2x2 factorial: model {vanilla, malaysian} x prompt {off, on}. Holding language on
  auto-detect for both keeps the only differences the two we are actually testing.
  `forced_en` was dropped after it made Whisper translate rather than transcribe.

* Corpus WER is total errors / total reference words, NOT the mean of per-clip WERs.
  Averaging per-clip rates over-weights short clips: a one-word error in a 4-word
  utterance would count as heavily as four errors in a 16-word one.

* Strict scoring counts `cek` vs `check` as an error. Lenient applies
  orthography_map.json first. Reporting both separates genuine mishearing from Whisper
  picking the other language's spelling -- the S5.6 limitation. Build the map FROM
  substitutions.csv rather than guessing it up front.

Usage
    python eval/speech/run_wer.py                    # transcribe + score
    python eval/speech/run_wer.py --score-only       # rescore cached transcripts
    python eval/speech/run_wer.py --configs vanilla_noprompt,malaysian_noprompt
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MANIFEST = HERE / "manifest.csv"
AUDIO_ROOT = HERE
ORTHO_MAP = HERE / "orthography_map.json"

# Deliberately shares no sentence or distinctive vocabulary with the test set -- see
# prompt_examples.txt. Overlap would let the model copy from its own context and
# inflate every number below.
#
# NOTE ON BALANCE: the first version of this prompt was almost entirely Malay, and it
# pushed the decoder so far toward Malay that English-dominant clips came back
# TRANSLATED ('how do i know' -> 'bagaimana saya tahu'). en_dom scored 71% WER as a
# result. A prompt for code-switched speech must itself be code-switched, with English
# words left in English orthography, or it biases one language at the other's expense.
ROJAK_PROMPT = (
    "cuaca panas gila hari ni kan, tak larat nak keluar. "
    "can you send me the file later, i need to check something first. "
    "eh awak dah reply that email ke belum lah. "
    "so how ah, do you want to join us or not."
)

VANILLA = "small"
MALAYSIAN = "mesolitica/malaysian-whisper-small-v3"

# `lang=None` means auto-detect per clip. Forcing a language is what makes Whisper
# translate instead of transcribe: forcing "en" on Malay audio produced English prose,
# and forcing "ms" on English-dominant clips produced Malay prose ('how do i know' ->
# 'bagaimana saya tahu'), which scored en_dom at 79% WER. For code-switched input the
# language must stay free, so both _auto variants exist to prove that on your own data.
CONFIGS = {
    "vanilla_noprompt":        dict(model="vanilla",   prompt=False, lang=None),
    "vanilla_prompt":          dict(model="vanilla",   prompt=True,  lang=None),
    "malaysian_noprompt":      dict(model="malaysian", prompt=False, lang="ms"),
    "malaysian_prompt":        dict(model="malaysian", prompt=True,  lang="ms"),
    "malaysian_prompt_auto":   dict(model="malaysian", prompt=True,  lang=None),
    "malaysian_noprompt_auto": dict(model="malaysian", prompt=False, lang=None),
}


# ---------------------------------------------------------------- transcription

_cache = {}


def get_vanilla():
    if "vanilla" not in _cache:
        import whisper
        print(f"loading openai-whisper '{VANILLA}' ...", flush=True)
        _cache["vanilla"] = whisper.load_model(VANILLA)
    return _cache["vanilla"]


def get_malaysian():
    if "malaysian" not in _cache:
        import torch
        from transformers import pipeline
        print(f"loading {MALAYSIAN} ...", flush=True)
        _cache["malaysian"] = pipeline(
            "automatic-speech-recognition",
            model=MALAYSIAN,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float32,
        )
    return _cache["malaysian"]


def transcribe(path: str, model: str, prompt: bool, lang):
    if model == "vanilla":
        out = get_vanilla().transcribe(
            path,
            language=lang,
            fp16=False,
            initial_prompt=ROJAK_PROMPT if prompt else None,
            # Clips are independent single utterances. Carrying context between them
            # would let one bad transcription bias the next.
            condition_on_previous_text=False,
        )
        return out["text"].strip(), out.get("language", "")

    asr = get_malaysian()
    gk = {"task": "transcribe"}
    # Mesolitica fine-tuned this checkpoint on Malay, so its generation_config pins the
    # language token. Simply omitting `language=` falls back to that default -- which is
    # why lang=None and lang="ms" produced byte-identical results on the first attempt.
    # Clearing forced_decoder_ids is what actually re-enables detection.
    gc = asr.model.generation_config
    if lang:
        gk["language"] = lang
        gc.forced_decoder_ids = getattr(gc, "_orig_forced", None) or gc.forced_decoder_ids
    else:
        if not hasattr(gc, "_orig_forced"):
            gc._orig_forced = getattr(gc, "forced_decoder_ids", None)
        gc.forced_decoder_ids = None
        gc.language = None
    if prompt:
        ids = asr.tokenizer.get_prompt_ids(ROJAK_PROMPT, return_tensors="pt")
        gk["prompt_ids"] = ids.to(asr.model.device)
    text = asr(path, generate_kwargs=gk)["text"].strip()
    text = _strip_prompt(text, ROJAK_PROMPT) if prompt else text
    return text, (lang or "auto")


def _strip_prompt(text: str, prompt: str) -> str:
    """Remove prompt text echoed into the transcript.

    HF echoes `prompt_ids` back into the output, sometimes whole, sometimes truncated,
    and not always at position 0. Left in, every echoed word scores as an insertion --
    which is how malaysian_prompt reached 106% WER with 215 insertions on a 387-word
    corpus. Checking only `startswith` was not enough.
    """
    t = " ".join(text.split())
    p = " ".join(prompt.split())
    low_t, low_p = t.lower(), p.lower()

    if low_p in low_t:                      # whole prompt echoed somewhere
        i = low_t.index(low_p)
        return (t[:i] + " " + t[i + len(p):]).strip()

    for k in range(len(p), 15, -5):         # truncated echo at the head
        if low_t.startswith(low_p[:k]):
            return t[k:].lstrip(" ,.-").strip()
    return t


# ---------------------------------------------------------------------- scoring

def build_transform():
    import jiwer
    return jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])


def apply_ortho(text: str, mapping: dict) -> str:
    """Canonicalise spelling variants before scoring.

    Lowercases and strips punctuation itself, because the jiwer transform runs *after*
    this and a map keyed on 'cek' would otherwise miss 'Cek' or 'cek,'.
    """
    out = []
    for w in text.split():
        k = w.lower().strip(".,!?;:")
        out.append(mapping.get(k, w))
    return " ".join(out)


def score_pair(ref: str, hyp: str, tf):
    """Return (errors, n_ref_words, substitution pairs)."""
    import jiwer
    out = jiwer.process_words(ref, hyp, reference_transform=tf, hypothesis_transform=tf)
    n = out.hits + out.substitutions + out.deletions
    errors = out.substitutions + out.deletions + out.insertions

    subs = []
    try:
        r_words, h_words = out.references[0], out.hypotheses[0]
        for ch in out.alignments[0]:
            if ch.type == "substitute":
                for a, b in zip(r_words[ch.ref_start_idx:ch.ref_end_idx],
                                h_words[ch.hyp_start_idx:ch.hyp_end_idx]):
                    subs.append((a, b))
    except Exception:
        pass  # alignment details are a bonus; never let them break scoring

    return dict(errors=errors, n=n, sub=out.substitutions,
                ins=out.insertions, dele=out.deletions), subs


def corpus_wer(df, col_err="errors", col_n="n"):
    """Total errors / total reference words -- not the mean of per-clip WERs."""
    n = df[col_n].sum()
    return df[col_err].sum() / n if n else float("nan")


# ------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="1")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--score-only", action="store_true",
                    help="rescore cached transcripts without re-running the models")
    args = ap.parse_args()

    chosen = [c.strip() for c in args.configs.split(",") if c.strip()]
    bad = [c for c in chosen if c not in CONFIGS]
    if bad:
        print(f"unknown config(s): {bad}\navailable: {list(CONFIGS)}")
        return 1

    man = pd.read_csv(MANIFEST)
    man = man[man["phase"].astype(str) == args.phase].reset_index(drop=True)
    print(f"manifest: {len(man)} clips (phase {args.phase})")

    missing = [p for p in man["audio_path"] if not (AUDIO_ROOT / p).exists()]
    if missing:
        print(f"\n{len(missing)} audio file(s) missing, e.g.:")
        for p in missing[:5]:
            print("  ", p)
        print("\nEvery clip must be present or the categories become unbalanced.")
        return 1
    print("all audio present\n")

    detail_path = HERE / "results_detail.csv"

    # ---- transcribe
    if args.score_only:
        if not detail_path.exists():
            print("--score-only needs results_detail.csv from an earlier run")
            return 1
        detail = pd.read_csv(detail_path)
        print(f"reusing {len(detail)} cached transcripts")
    else:
        rows = []
        for cfg in chosen:
            kw = CONFIGS[cfg]
            print(f"--- {cfg} ---", flush=True)
            for i, r in enumerate(man.itertuples(), 1):
                hyp, det = transcribe(str(AUDIO_ROOT / r.audio_path), **kw)
                rows.append(dict(config=cfg, id=r.id, switch_type=r.switch_type,
                                 speaker=r.speaker, reference=r.reference, hypothesis=hyp,
                                 detected_lang=det))
                if i % 8 == 0:
                    print(f"  {i}/{len(man)}", flush=True)
        detail = pd.DataFrame(rows)
        detail.to_csv(detail_path, index=False)
        print(f"\nwrote {detail_path.name}")

    # ---- score
    tf = build_transform()
    mapping = json.loads(ORTHO_MAP.read_text(encoding="utf-8")) if ORTHO_MAP.exists() else {}
    mapping = {k: v for k, v in mapping.items() if not k.startswith("_")}
    if mapping:
        print(f"orthography map: {len(mapping)} entries (lenient scoring enabled)")
    else:
        print("no orthography_map.json -- lenient == strict for now")

    all_subs = Counter()
    recs = []
    for r in detail.itertuples():
        strict, subs = score_pair(r.reference, r.hypothesis, tf)
        lenient, _ = score_pair(apply_ortho(r.reference, mapping),
                                apply_ortho(r.hypothesis, mapping), tf)
        all_subs.update(subs)
        recs.append(dict(config=r.config, id=r.id, switch_type=r.switch_type,
                         speaker=r.speaker, reference=r.reference, hypothesis=r.hypothesis,
                         detected_lang=getattr(r, "detected_lang", ""),
                         errors=strict["errors"], n=strict["n"], sub=strict["sub"],
                         ins=strict["ins"], dele=strict["dele"],
                         errors_lenient=lenient["errors"], n_lenient=lenient["n"]))
    scored = pd.DataFrame(recs)
    scored.to_csv(detail_path, index=False)

    # ---- summary
    summary = []
    for cfg, g in scored.groupby("config"):
        summary.append(dict(
            config=cfg, clips=len(g), ref_words=int(g["n"].sum()),
            wer_strict=round(corpus_wer(g), 4),
            wer_lenient=round(corpus_wer(g, "errors_lenient", "n_lenient"), 4),
            sub=int(g["sub"].sum()), ins=int(g["ins"].sum()), dele=int(g["dele"].sum()),
        ))
    summary = pd.DataFrame(summary).sort_values("wer_strict").reset_index(drop=True)
    summary.to_csv(HERE / "results_summary.csv", index=False)

    cat = (scored.groupby(["config", "switch_type"])
           .apply(lambda g: pd.Series({"clips": len(g), "ref_words": int(g["n"].sum()),
                                       "wer_strict": round(corpus_wer(g), 4)}),
                  include_groups=False)
           .reset_index())
    cat.to_csv(HERE / "results_by_category.csv", index=False)

    pd.DataFrame([{"reference_word": a, "whisper_wrote": b, "count": c}
                  for (a, b), c in all_subs.most_common()]
                 ).to_csv(HERE / "substitutions.csv", index=False)

    # ---- report
    pd.set_option("display.width", 200)
    print("\n" + "=" * 72)
    print("TABLE A -- WER by configuration")
    print("=" * 72)
    print(summary.to_string(index=False))

    print("\n" + "=" * 72)
    print("TABLE B -- WER by switch type (best config)")
    print("=" * 72)
    best = summary.iloc[0]["config"]
    print(f"config: {best}")
    print(cat[cat["config"] == best].sort_values("wer_strict").to_string(index=False))

    print("\nTop substitutions (build orthography_map.json from these):")
    for (a, b), c in all_subs.most_common(15):
        print(f"  {c:>3}x  {a!r} -> {b!r}")

    print(f"\nwrote results_summary.csv, results_by_category.csv, substitutions.csv")
    print(f"target: WER <= 0.25 (proposal S7.4). Best here: {summary.iloc[0]['wer_strict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
