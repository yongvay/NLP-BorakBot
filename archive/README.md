# archive/ — tooling that has already run

Nothing in here is imported by the running application. `app/streamlit_app.py` imports
exactly one first-party module, `app/stt.py`, and that is the whole live import graph.

These scripts were each run once, produced a committed artefact, and are kept because that
artefact is cited in the report and in the academic-integrity declaration. Keeping them
runnable is the point; keeping them in the way was not.

| Subtree | What it did | Its committed output |
|---|---|---|
| `eval/speech/run_wer.py` | Swept 9 decode configurations over 48 clips and scored WER. Produced the **0.349 strict / 0.339 lenient** figure quoted in the report and in the app sidebar. | `eval/speech/results/*.csv` |
| `eval/speech/threshold_sweep.py` | Reconstructed routed WER at 21 English thresholds from the cached transcripts, without re-decoding. Justifies `EN_THRESHOLD = 0.5`. | `eval/speech/results/threshold_sweep.csv` |
| `eval/speech/inspect_results.py` | Ad-hoc filtering of `results_detail.csv` while writing up. | — |
| `scripts/prepare_audio.py` | Converted and renamed 48 phone recordings into the manifest's filenames. | `eval/speech/audio/{yv,pq}/*.wav` |
| `scripts/build_corpus.py` | Assembled the training corpus from the raw generation batches, dropping 16 cross-fact templates and 14 near-duplicates. | `data/generated/pairs_v1.jsonl` (632 pairs) |
| `scripts/validate_pairs.py` | Gate on the corpus — emoji, sentence count, refusal coverage. Exits non-zero on failure. | — (pass/fail only) |
| `scripts/prompts/generate_pairs_prompt.md` | The prompt used to generate the training pairs. **This is the provenance record for the AI-generated-data declaration.** | `data/generated/raw/*.jsonl` |
| `notebooks/run_speech_eval.ipynb` | Thin Colab wrapper that ran `run_wer.py` on a GPU. | — |

## Running them again

The eval scripts need `jiwer`, which `requirements.txt` deliberately leaves out of the app
install:

```
pip install jiwer
python archive/eval/speech/run_wer.py --score-only     # rescore from cached transcripts
python archive/eval/speech/run_wer.py                  # full re-decode, 432 transcriptions
```

The corpus scripts resolve `data/` relative to the working directory, so run them **from the
repository root**:

```
python archive/scripts/validate_pairs.py --dir data/generated/raw
python archive/scripts/build_corpus.py --dry-run
```

`build_corpus.py` reads `data/generated/raw/`, which is gitignored. If you cloned this
repository fresh, that directory is empty and the script has nothing to rebuild from —
`data/generated/pairs_v1.jsonl` is the committed result.

## Deleted rather than archived

Three files were removed outright in the same change, recoverable from git history:

- `scripts/verify_stt_matches_eval.py` — a second copy of the drift guard that
  `app/stt.py --check-config` already performed. Both are gone; the app and the harness are
  frozen, and the constants they share are marked as such in `app/stt.py`.
- `eval/speech/sync_report_figures.py` — rewrote tables in `docs/PartB_Draft_STT.docx`. It
  required `python-docx`, which is not in the pinned environment, so it could not run.
- `notebooks/select_whisper.ipynb` — model-selection exploration, superseded by `run_wer.py`.
  It still contained the Malay-only prompt that the report explicitly records as rejected.
