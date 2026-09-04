"""Look at the transcripts behind the numbers. Run after run_wer.py.

NOT named inspect.py -- that shadows the stdlib `inspect` module, which pandas imports
internally, and breaks every import of pandas from this directory.

A WER table tells you something is wrong; only the transcripts tell you what.

    python stage1_stt/inspect_results.py                     # worst clips, best config
    python stage1_stt/inspect_results.py --config vanilla_prompt --category en_dom
"""
import argparse
import pandas as pd

from run_wer import find_result   # resolves a fresh run or the committed copy

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--category")
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--langs", action="store_true",
                help="detected language per category instead of transcripts")
a = ap.parse_args()

d = pd.read_csv(find_result("results_detail.csv"))
d["wer"] = d["errors"] / d["n"]

cfg = a.config or (pd.read_csv(find_result("results_summary.csv")).iloc[0]["config"])
d = d[d["config"] == cfg]
if a.category:
    d = d[d["switch_type"] == a.category]

print(f"config: {cfg}" + (f"   category: {a.category}" if a.category else ""))
print(f"clips: {len(d)}   corpus WER: {d['errors'].sum() / d['n'].sum():.3f}\n")

if a.langs:
    if "detected_lang" not in d.columns:
        print("no detected_lang column -- re-run run_wer.py")
        raise SystemExit(0)
    print(pd.crosstab(d["switch_type"], d["detected_lang"]).to_string())
    print("\n'ms' on an en_dom row means Whisper chose to write Malay for English audio.")
    raise SystemExit(0)

for r in d.sort_values("wer", ascending=False).head(a.n).itertuples():
    print(f"[{r.id}] {r.switch_type}  WER={r.wer:.2f}  (S{r.sub} I{r.ins} D{r.dele})")
    print(f"  said : {r.reference}")
    print(f"  got  : {r.hypothesis}")
    print()
