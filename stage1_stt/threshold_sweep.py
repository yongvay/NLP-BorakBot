"""Sensitivity of routed WER to the English-detection threshold.

Answers one question the main results cannot: is `en_dom` weak because the detector
cannot hear the English, or because the cutoff is set too high?

    detector confident but below cutoff  ->  the constant is wrong, and this is cheap
    detector not confident at all        ->  routing has hit its ceiling

HOW IT AVOIDS A GPU RE-RUN
--------------------------
Routing only ever picks one of two language tokens, and both branches have already been
transcribed for every clip: `malaysian_prompt` is the forced-Malay branch and
`malaysian_prompt_en` the forced-English one. So the routed transcript at any threshold
is a *selection* over transcripts that already exist:

    hypothesis(clip, t) = forced_en[clip]  if p_en[clip] > t  else  forced_ms[clip]

Nothing is re-decoded. The sweep is a scoring pass over cached text.

SELF-CHECK
----------
At t = EN_THRESHOLD the reconstruction must reproduce the measured WER of
`malaysian_prompt_routed` to within rounding. If it does not, the reconstruction is
invalid and every other row here is meaningless -- so the script checks this first and
refuses to continue if it fails. A validation that cannot fail proves nothing; this one
can, and that is the point.

ON READING THESE NUMBERS
------------------------
This is a sensitivity analysis, NOT a tuning procedure. Picking the minimising threshold
and reporting the resulting WER as a headline result would be fitting a decision rule to
the same 48 clips it is scored on, and the corpus is too small to hold out a tuning
split. Report the curve and the a priori choice; do not move the choice to the minimum.

Usage
    python stage1_stt/run_wer.py --detect-only     # writes detector_probs.csv
    python stage1_stt/threshold_sweep.py
"""

import sys
from pathlib import Path

import pandas as pd

from run_wer import (EN_THRESHOLD, ORTHO_MAP, apply_ortho, build_transform,
                     corpus_wer, find_result, score_pair)

HERE = Path(__file__).resolve().parent

MS_CONFIG = "malaysian_prompt"        # forced-Malay branch
EN_CONFIG = "malaysian_prompt_en"     # forced-English branch
ROUTED_CONFIG = "malaysian_prompt_routed"   # the measured result, for the self-check

GRID = [round(x * 0.05, 2) for x in range(0, 21)]   # 0.00 .. 1.00


def main() -> int:
    probs = pd.read_csv(find_result("detector_probs.csv")).set_index("id")
    detail = pd.read_csv(find_result("results_detail.csv"))

    have = set(detail.config.unique())
    for c in (MS_CONFIG, EN_CONFIG, ROUTED_CONFIG):
        if c not in have:
            sys.exit(f"results_detail.csv has no rows for {c!r}; run it first")

    ms = detail[detail.config == MS_CONFIG].set_index("id")
    en = detail[detail.config == EN_CONFIG].set_index("id")
    routed = detail[detail.config == ROUTED_CONFIG]

    missing = set(ms.index) - set(probs.index)
    if missing:
        sys.exit(f"{len(missing)} clips have no p_en; re-run --detect-only")

    tf = build_transform()
    mapping = {}
    if ORTHO_MAP.exists():
        import json
        mapping = {k: v for k, v in json.loads(ORTHO_MAP.read_text(encoding="utf-8")).items()
                   if not k.startswith("_")}

    def evaluate(t: float) -> pd.DataFrame:
        """Score the corpus with the language chosen by `p_en > t`."""
        recs = []
        for cid, row in ms.iterrows():
            use_en = probs.loc[cid, "p_en"] > t
            hyp = en.loc[cid, "hypothesis"] if use_en else row["hypothesis"]
            hyp = "" if pd.isna(hyp) else str(hyp)
            strict, _ = score_pair(row["reference"], hyp, tf)
            lenient, _ = score_pair(apply_ortho(row["reference"], mapping),
                                    apply_ortho(hyp, mapping), tf)
            recs.append(dict(id=cid, switch_type=row["switch_type"], chose="en" if use_en else "ms",
                             errors=strict["errors"], n=strict["n"],
                             errors_lenient=lenient["errors"], n_lenient=lenient["n"]))
        return pd.DataFrame(recs)

    # ---- self-check: reproduce the measured routed result at the a priori threshold
    # p_en > 0.5 implies English is the top language, so at t = 0.5 this rule is
    # identical to the one in detect_lang. Below 0.5 they diverge (detect_lang also
    # requires English to be the argmax); the sweep uses the simpler rule and says so.
    rebuilt = corpus_wer(evaluate(EN_THRESHOLD))
    measured = corpus_wer(routed)
    delta = abs(rebuilt - measured)
    print(f"self-check at t={EN_THRESHOLD}: rebuilt {rebuilt:.4f} vs measured {measured:.4f} "
          f"(delta {delta:.4f})")
    if delta > 0.005:
        print("\nFAIL -- the reconstruction does not reproduce the measured result.")
        print("The cached branches and detector probabilities are inconsistent; the")
        print("sweep below would be meaningless. Re-run --detect-only and the two")
        print("configs together before trusting anything here.")
        return 1
    print("ok -- reconstruction is faithful\n")

    # ---- sweep
    rows, per_cat = [], {}
    for t in GRID:
        d = evaluate(t)
        rows.append(dict(threshold=t, n_en=int((d.chose == "en").sum()),
                         wer_strict=round(corpus_wer(d), 4),
                         wer_lenient=round(corpus_wer(d, "errors_lenient", "n_lenient"), 4)))
        per_cat[t] = d.groupby("switch_type").apply(
            lambda g: round(corpus_wer(g), 4), include_groups=False)

    sweep = pd.DataFrame(rows)
    sweep.to_csv(HERE / "threshold_sweep.csv", index=False)

    print("=" * 64)
    print("THRESHOLD SWEEP -- routed WER as a function of the English cutoff")
    print("=" * 64)
    marked = sweep.copy()
    marked["note"] = ["  <- a priori choice, reported" if t == EN_THRESHOLD else ""
                      for t in marked.threshold]
    print(marked.to_string(index=False))

    best = sweep.loc[sweep.wer_strict.idxmin()]
    print(f"\nminimum on this corpus: {best.wer_strict:.4f} at t={best.threshold} "
          f"({int(best.n_en)}/48 clips routed to English)")
    print(f"reported result:        {rebuilt:.4f} at t={EN_THRESHOLD}")
    print(f"headroom:               {rebuilt - best.wer_strict:.4f}")
    print("\nThe minimum is a ceiling, not a result. Adopting it would fit the rule to")
    print("the test set. Report the curve; keep the a priori threshold.")

    print("\n" + "=" * 64)
    print("BY CATEGORY -- where the threshold actually bites")
    print("=" * 64)
    cat = pd.DataFrame(per_cat).T
    cat.index.name = "threshold"
    print(cat.to_string())

    print(f"\nwrote threshold_sweep.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
