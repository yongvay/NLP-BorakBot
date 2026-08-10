"""Look at the transcripts behind the numbers. Run after run_wer.py.

NOT named inspect.py -- that shadows the stdlib `inspect` module, which pandas imports
internally, and breaks every import of pandas from this directory.

A WER table tells you something is wrong; only the transcripts tell you what.

    python eval/speech/inspect_results.py                     # worst clips, best config
    python eval/speech/inspect_results.py --config vanilla_prompt --category en_dom
"""
import argparse
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--config")
ap.add_argument("--category")
ap.add_argument("--n", type=int, default=12)
a = ap.parse_args()

d = pd.read_csv(HERE / "results_detail.csv")
d["wer"] = d["errors"] / d["n"]

cfg = a.config or (pd.read_csv(HERE / "results_summary.csv").iloc[0]["config"])
d = d[d["config"] == cfg]
if a.category:
    d = d[d["switch_type"] == a.category]

print(f"config: {cfg}" + (f"   category: {a.category}" if a.category else ""))
print(f"clips: {len(d)}   corpus WER: {d['errors'].sum() / d['n'].sum():.3f}\n")

for r in d.sort_values("wer", ascending=False).head(a.n).itertuples():
    print(f"[{r.id}] {r.switch_type}  WER={r.wer:.2f}  (S{r.sub} I{r.ins} D{r.dele})")
    print(f"  said : {r.reference}")
    print(f"  got  : {r.hypothesis}")
    print()
