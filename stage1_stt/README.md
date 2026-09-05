# Stage 1 — Speech to text

The Whisper half of BorakBot: the test set, the WER harness, and the committed results
behind every speech figure in the report.

This is **evidence, not dead code.** The numbers produced here are quoted in the report
and in the app sidebar, and the constants they justify are live in `app/stt.py`:

| Number | Where it came from | Where it is used |
|---|---|---|
| WER **0.349** strict / **0.339** lenient | `run_wer.py`, 9 decode configs over 48 clips | report, app sidebar |
| `EN_THRESHOLD = 0.5` | `threshold_sweep.py`, 21 thresholds | `app/stt.py` |
| `ASR_REPAIRS` rules | `results/substitutions.csv`, 492 rows | `app/normalise.py` |

Full method and rationale: the Part B report.

## What is here

| File | What it did | Its committed output |
|---|---|---|
| `run_wer.py` | Swept 9 decode configurations over 48 clips and scored WER. | `results/*.csv` |
| `threshold_sweep.py` | Reconstructed routed WER at 21 English thresholds from the cached transcripts, without re-decoding. Justifies `EN_THRESHOLD = 0.5`. | `results/threshold_sweep.csv` |
| `inspect_results.py` | Ad-hoc filtering of `results_detail.csv` while writing up. | — |
| `prepare_audio.py` | Converted and renamed 48 phone recordings into the manifest's filenames. | `audio/{yv,pq}/*.wav` |
| `run_speech_eval.ipynb` | Thin Colab wrapper that ran `run_wer.py` on a GPU. | — |
| `manifest.csv` | 96 rows: id, audio_path, reference, switch_type, speaker, phase. **The source of truth.** | — |
| `orthography_map.json` | Spelling variants treated as equivalent for lenient WER. | — |
| `results/` | Six CSVs. Committed. | — |

Nothing here is imported by the running application. `app/stt.py` deliberately keeps its own
copy of the shared constants rather than importing this folder, so that `pandas` and `jiwer`
stay out of the deployment; the duplication is marked in `app/stt.py`.

## Running it again

The harness needs `jiwer`, which `requirements.txt` deliberately leaves out of the app install:

```
pip install jiwer
python stage1_stt/run_wer.py --score-only     # rescore from cached transcripts
python stage1_stt/run_wer.py                  # full re-decode, 432 transcriptions
python stage1_stt/threshold_sweep.py
python stage1_stt/inspect_results.py --category en_dom
```

**Output paths.** A run writes its CSVs *beside* `run_wer.py`; the copies committed to the
repo live in `results/`. `find_result()` in `run_wer.py` prefers a fresh run and falls back
to the committed copy, so readers work either way — but if the two disagree, the committed
copies are stale. That divergence is what let three figures in an earlier report draft drift
away from the CSVs backing them.

## Recording method

1. Read the `reference` column aloud at normal pace. Do NOT over-enunciate — you are
   measuring performance on realistic input.
2. Phone voice memo is fine. Quiet room.
3. Convert to the format Whisper wants (16 kHz mono WAV):

   ```
   ffmpeg -i raw/s01.m4a -ar 16000 -ac 1 stage1_stt/audio/yv/s01_ms_dom_yv.wav
   ```

   Or let `prepare_audio.py` do the renaming, which is where hand-renaming 24 files goes wrong.
4. Track progress in `recording_tracker_UPDATED.xlsx` (yellow cells only). `manifest.csv` holds
   the data and is what the harness reads — never edit sentences in Excel.

References were written BEFORE recording, deliberately. Transcribing your own audio afterwards
biases the reference toward whatever you happened to say.

## Stratification

| switch_type | n | tests |
|---|---|---|
| ms_dom    | 8 | Malay matrix, English insertion |
| en_dom    | 8 | English matrix, Malay particles |
| balanced  | 8 | true intra-sentential switching |
| particles | 8 | discourse particles (lah/lor/meh/kan/kot) |
| numeric   | 8 | prices, percentages, dates |
| entity    | 8 | Malaysian named entities |

Proposal S5.6 predicts `balanced` and `particles` score worst, because Whisper commits to one
language per segment. Confirming that with our own data is the finding — not a failure.

## Git

Commit `manifest.csv`, `prompt_examples.txt`, `results/*.csv`.
Do NOT commit `audio/` or `raw/` (see `.gitignore`). Audio lives in the shared Drive.

## Deleted rather than archived

Three Stage 1 files were removed outright, recoverable from git history:

- `scripts/verify_stt_matches_eval.py` — a second copy of the drift guard that
  `app/stt.py --check-config` already performed. Both are gone; the app and the harness are
  frozen, and the constants they share are marked as such in `app/stt.py`.
- `eval/speech/sync_report_figures.py` — rewrote tables in the Stage 1 Word draft. It required
  `python-docx`, which is not in the pinned environment, so it could not run.
- `notebooks/select_whisper.ipynb` — model-selection exploration, superseded by `run_wer.py`.
  It still contained the Malay-only prompt that the report explicitly records as rejected.
