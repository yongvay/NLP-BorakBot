# BorakBot — Handoff 1 (SUPERSEDED)

> ⚠ **Read `HANDOFF_2.md` instead.** This file is kept for history. Where the two
> disagree, Handoff 2 is right — it corrects two claims made here, listed in its §3.
> The status tables, next steps and open decisions below are all out of date.

Written 10 August 2026. Read this first in a new session, then `CLAUDE.md` for the
project rules.

---

## 1. Where the project stands

**Deadline: 31 August 2026.** 21 days left. Graded on a live demo with individual Q&A
(rubric item 11), so both team members must be able to explain every design decision.

| Module | Status |
|---|---|
| **Speech-to-text (Stage 1)** | **Done** — built, evaluated, written up |
| Rojak normalisation (Stage 2) | Not started |
| Knowledge base | Scoped; 28 of ~165 facts drafted |
| Synthetic rojak dataset | Not started — generator not written |
| QLoRA fine-tuning (Stage 3) | Not started |
| Streamlit UI | Not started |
| Feedback loop (Stage 4) | Not started |

The dataset gates fine-tuning, which gates the UI, evaluation and demo. **It is the
critical path.** Do not spend more time on STT beyond `app/stt.py` and the threshold
sensitivity run, both of which are small and already specified below.

---

## 2. STT: what was built

A two-model pipeline. Both models are Whisper — the second is a fine-tune of the first,
same architecture, same 244M parameters.

| Role | Model | Job |
|---|---|---|
| Detector | `openai/whisper-small` | one encoder pass → decides `en` or `ms` |
| Transcriber | `mesolitica/malaysian-whisper-small-v3` | writes the text using that language token |

Plus a short **code-switched prompt** supplied as decoder context.

### Why routing exists

Whisper's decoder starts with `<|startoftranscript|> <|LANG|> <|transcribe|>`. The
middle token specifies **what language to write in**, not what was heard. Mismatch it
and the model translates instead of transcribing.

The Mesolitica checkpoint pins `<|ms|>` in its `generation_config`, so English clips
came back as fluent Malay (`en_dom` WER 0.788). Setting `language=None` changed nothing
— results were byte-identical, which is how the pinning was discovered. Clearing
`forced_decoder_ids` re-enables detection, but Mesolitica's own detector still always
answers Malay. Hence: vanilla detects, Mesolitica transcribes.

The threshold is **asymmetric** — English is only selected when `P(en) > 0.5` — because
forcing English on Malay audio is catastrophic (6.483 WER, runaway repetition) while the
reverse merely translates (0.788).

> ⚠ **CORRECTED.** That 6.483 is one clip, not a category-wide effect: seven of the
> eight `ms_dom` clips transcribe fine under forced English (0, 0, 0, 0, 1, 1, 5
> errors) and `s05_pq` alone contributes 362 of the 379 insertions. Forcing the wrong
> language is *usually harmless and occasionally unbounded*. The threshold is still
> right; the justification is expected cost under a heavy tail. See `HANDOFF_2.md` §3.

---

## 3. Results (final, from the last Colab run)

**Test set**: 48 utterances, 387 reference words, 6 categories × 8, two speakers,
4 per speaker per category so speaker is never confounded with category.

### Table A — WER by configuration

Corrected 10 August 2026 against `results_summary.csv`. The three vanilla rows had
drifted; they are now regenerated from the CSV by `sync_report_figures.py`.

| Configuration | strict | lenient |
|---|---|---|
| **malaysian_prompt_routed** | **0.349** | **0.339** |
| malaysian_prompt_auto | 0.398 | 0.390 |
| malaysian_prompt (forced ms) | 0.398 | 0.390 |
| vanilla_prompt_routed | 0.429 | 0.419 |
| vanilla_prompt | 0.447 | 0.439 |
| vanilla_noprompt | 0.592 | 0.581 |
| malaysian_noprompt_auto | 0.600 | 0.597 |
| malaysian_noprompt (forced ms) | 0.618 | 0.615 |
| malaysian_prompt_en *(diagnostic only)* | 1.274 | 1.264 |

### Table B — WER by switch type, best configuration

| Category | ref words | WER |
|---|---|---|
| ms_dom | 58 | 0.155 |
| particles | 58 | 0.190 |
| balanced | 64 | 0.234 |
| entity | 60 | 0.367 |
| en_dom | 66 | 0.530 |
| numeric | 81 | 0.531 |
| **Overall** | **387** | **0.349** |

### What the numbers mean

- **Target missed**: proposal §7.4 set WER ≤ 0.25; best is 0.349. Reported honestly.
  The target was set before any measurement and is optimistic for a small-class model on
  code-switched speech (published figures typically 0.30–0.50).
- **Relative improvement is the real claim**: 0.618 → 0.349, a **44% reduction**.
  Prompting contributed 22.0 points, the Malaysian fine-tune 4.9, routing 4.9.
  (Recomputed from the CSV — the fine-tune figure was previously stated as ~6.)
  Prompting is worth 22 points on the Malaysian checkpoint but only 14.5 on vanilla,
  so "22 points on both models" was wrong and has been corrected in the report.
- **Excluding `numeric` and `en_dom`**, the remaining 240 words score **0.238** — under
  target. ⚠ **CORRECTED:** these two exclusions are not equivalent. `numeric` is a
  scoring artefact arising outside the system; `en_dom` is a failure of our own routing
  component. Excluding both as "external causes" flatters the result.
- **Oracle bound** (best language per category) is **0.277**. That caps what routing
  alone can achieve here.
- **The §5.6 prediction failed.** Intra-sentential switching (`balanced`, 0.234) was
  predicted to be worst; it came third. Written up as a stated non-confirmation — do not
  quietly drop it.
- `numeric` is mostly representational: references are written as spoken
  (`thirty one`) while Whisper writes digits (`31`). A word-level map cannot fix
  many-to-one conversions; a number normaliser would be needed.
- `entity` errors are genuine (`MyJPJ` → `jbj`). Adding entity names to the prompt would
  help but is **leakage**, since they appear in the test set. Mention as future work only.

### ✅ Data-integrity task — resolved 10 August 2026

The 9-config run had in fact already been done: `results_detail.csv` holds all 432 rows
(48 clips × 9 configs) and all four CSVs are committed. **Nothing needs re-running.**

The prediction that figures would shift was correct, and it hit the three vanilla rows.
The report has been corrected and Table A / Table B are now *generated* from the CSVs by
`eval/speech/sync_report_figures.py` rather than transcribed by hand — run it after
every evaluation, and it will refuse to let the two drift again:

```
python eval/speech/sync_report_figures.py           # check
python eval/speech/sync_report_figures.py --write   # rewrite tables from CSV
```

Root cause worth remembering: `run_wer.py` writes results to `eval/speech/`, notebook
Cell 3 copies them to Drive, and the committed copies live in `eval/speech/results/`.
Three locations, manual copying between them. If they disagree, the committed copies
are the stale ones.

### ⚠ New finding — the router under-selects English

Only **3 of 8 `en_dom` clips** are routed to English. On that category routing makes 35
errors where forced-English makes 7, and four clips (s09, s11, s12, s14) score **zero**
errors under forced English while routing sends them to Malay. So `en_dom` at 0.530 is a
*routing* failure, not a transcription failure — a sharper claim than the report
currently makes.

The asymmetric threshold is still justified: the 6.483 catastrophe is entirely `ms_dom`
(362 of 379 runaway insertions land there). The cutoff is simply conservative.

**Do not tune it on this test set** — 48 clips with no holdout, and tuning a decision
rule on the data it is scored on is leakage, the same principle that kept entity names
out of the prompt. Instead:

```
python eval/speech/run_wer.py --detect-only    # logs p_en per clip, no GPU
python eval/speech/threshold_sweep.py          # WER vs threshold, from cached transcripts
```

`p_en` distinguishes *missed narrowly* (cutoff wrong, cheap fix) from *missed
completely* (detector deaf, routing has hit its ceiling) — the current CSVs cannot tell
these apart, and they support opposite conclusions. Report the curve as a sensitivity
analysis and keep the a priori 0.5. The sweep self-checks by reconstructing the measured
0.349 at t=0.5 and aborts if it cannot.

---

## 4. Repository map

```
CLAUDE.md                            project rules — read after this file
HANDOFF.md                           this file
docs/
  NLP Assignment Part A.md/.pdf      the proposal
  Rubrics.pdf, Specification.pdf     grading criteria
  BMDS2123 ... CoverPage.docx.md     Part B template
  whisper_step_by_step.md            STT methodology guide
  PartB_Draft_STT.docx               report draft (yellow = blanks to fill)
notebooks/
  run_speech_eval.ipynb              THE Colab notebook — cells in order
  try_whisper.ipynb                  first exploration, superseded
scripts/
  prepare_audio.py                   phone recordings → 16 kHz mono WAV, renamed
eval/speech/
  manifest.csv                       48 utterances × 2 speakers, phase 1/2
  recording_tracker_UPDATED.xlsx     per-speaker recording checklist
  run_wer.py                         transcription + WER across configs
  inspect_results.py                 transcript inspection (NOT inspect.py — shadows stdlib)
  threshold_sweep.py                 routed WER vs detection threshold, no GPU
  sync_report_figures.py             regenerates the report's tables from the CSVs
  orthography_map.json               documented spelling variants for lenient scoring
  prompt_examples.txt                prompt candidates, disjoint from the test set
  results/                           the four committed result CSVs
data/knowledge_base/
  README.md                          fact schema and rules for writing facts
  _scope.md                          domain boundary + the Part A deviation writeup
  _system_prompt.md                  the constrained system prompt (Part A §6.1 Stage 3)
  _out_of_scope.yaml                 23 probes that must trigger the fallback
  jpj_vehicle_licence.yaml           18 facts
  jpn_identity.yaml                  5 facts
  immigration_passport.yaml          5 facts
```

Not yet written: `app/`, `training/`, `scripts/generate_pairs.py`, and four more
knowledge-base domains (`epf_socso`, `lhdn_tax`, `slang_meanings`, `small_talk`).

---

## 5. Workflow (this trips people up repeatedly)

```
code    lives on GitHub   →  Colab gets it via git clone / git pull
audio   lives on Drive    →  copied into Colab by Cell 1
results written in /content →  Cell 3 copies them back to Drive
```

`/content` is **scratch** and is erased on every runtime restart. That is normal.

After editing code locally you **must** `git add -A && git commit && git push`, then
`!git pull` in Colab. Colab cannot see the laptop.

- GitHub: `https://github.com/yongvay/NLP-BorakBot`
- Drive audio: `MyDrive/RDS3S1/NLP/NLP Assignment/Audio`
- Drive results: `MyDrive/RDS3S1/NLP/NLP Assignment/results`
- Audio is gitignored (`.wav`, `.mp4`, `.m4a`, `eval/speech/audio/`, `eval/speech/raw/`)

---

## 6. Deviations from the Part A proposal

State these in Part B rather than letting them surface at the demo.

1. **STT moved from Week 4 to first**, on the tutor's advice. Justified: it is the only
   module with no dependency on the fine-tuned LLM.
2. **A Malaysian-fine-tuned Whisper** (`mesolitica/malaysian-whisper-small-v3`) replaces
   plain `whisper-small` for transcription. Same architecture; directly addresses the
   §5.6 limitation. Mesolitica was already cited in Part A for MaLLaM.
3. **Language routing** added — not described in Part A, justified empirically.
4. **The §7.4 WER target was not met** (0.349 vs 0.25).
5. **Timeline compressed** — the 7-week Gantt is being executed in ~3 weeks.

---

## 7. Next steps, in order

1. ~~One clean 9-config run~~ — **already done**; see §3. Report figures corrected.
2. **Threshold sensitivity run** (Colab Cell 5, ~10 min, no GPU): `--detect-only` then
   `threshold_sweep.py`. Turns `en_dom` from an unexplained weak category into a
   measured routing limitation. Write the result into Part B §5.
3. **Write `app/stt.py`** — package routing as `transcribe(path) -> str` for Streamlit.
   Not yet written; the report's Coding section references it as pending.
4. ~~Define the knowledge base scope~~ — **decided 10 Aug**: Malaysian government
   services and personal admin, plus a thin conversational layer. This narrows Part A §4
   and is deviation #6; the justification is written in `data/knowledge_base/_scope.md`.
5. **Finish the knowledge base** (4 domains left, ~137 facts), write
   `scripts/generate_pairs.py`, generate ~1,650 rojak Q&A pairs, then human-validate a
   sample and report agreement statistics (proposal §7.3). Build the generator *before*
   writing the remaining facts — if one fact does not expand into ten good pairs, the
   fix is a schema change, and that is cheap at 28 facts and expensive at 165.
5. **QLoRA fine-tune** round 1 on Kaggle (LLaMA-Factory, 3B-class instruct base, 4-bit,
   T4). Kaggle's 30 GPU-h/week is for this — don't burn it on Colab-scale work.
6. **Baseline comparison**: fine-tuned vs un-fine-tuned on perplexity, BLEU, ROUGE-L,
   BERTScore, plus human Likert ratings.
7. **Streamlit UI + SQLite feedback logging**, then corrections → retrain round 2.
8. **Report and demo prep.** Both members must be able to defend every decision.

---

## 8. Open decisions the team must make

1. ~~Knowledge base scope~~ — **decided**, see `data/knowledge_base/_scope.md`.
2. **How to generate the synthetic dataset** — in-session with an assistant, via a paid
   API with a reproducible script, or with an open model on Kaggle/Colab. Whichever is
   chosen must be reproducible and declared in Appendix A.
3. **Contribution split** for the cover page (currently drafted 50/50).
4. **Programme name and tutor name** for the cover page.
5. **Appendix A AI-assistance declaration** — required by TARUMT policy. Must be written
   by the team, honestly and specifically.

---

## 9. Gotchas already hit — don't repeat them

- `inspect.py` shadows Python's stdlib `inspect`, which pandas imports. Hence
  `inspect_results.py`.
- `run_wer.py --configs <subset>` used to overwrite all other configs' cached
  transcripts. Now merges — but check `results_detail.csv` row count (48 × n configs).
- `prepare_audio.py` pairs raw files to manifest rows **by position**. Copying off a
  phone rewrites modification times, so use `--by-name` (numeric-aware sort). It refuses
  to run when timestamp and filename order disagree.
- A validation that cannot fail is not a validation: the eval preview's `output` and
  `reference` columns both come from the manifest, so they always agree. Only the raw
  filename column proves anything.
- The prompt must be **code-switched**. An all-Malay prompt pushed English clips into
  translation (`en_dom` 0.788).
- Prompt text must share **no content with the test set**, or the model copies from its
  own context and every figure is inflated.
- Colab drops the runtime after ~90 min idle. Always re-run Cell 1 first.
