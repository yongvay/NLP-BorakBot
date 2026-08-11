# BorakBot — Handoff 2

Written 11 August 2026. **This is the current handoff.** Read it first, then
`CLAUDE.md` for the project rules. `HANDOFF_1.md` is kept for history; where the two
disagree, this one is right.

---

## 1. Where the project stands

**Deadline: 31 August 2026 — 20 days.** Graded on a live demo with individual Q&A
(rubric item 11), so both members must be able to explain every design decision in
this repository.

| Module | Status |
|---|---|
| **Speech-to-text (Stage 1)** | **Closed** — built, evaluated, written up, packaged |
| **Streamlit UI shell** | **Stage 1 only** — records and transcribes, nothing else. Verified end to end on Windows, 11 Aug: ~6.6 s/utterance on CPU |
| Knowledge base | Scoped; **28 of ~165 facts** |
| Synthetic dataset generator | **Not written** ← the next thing to do |
| Rojak normalisation (Stage 2) | Not started |
| QLoRA fine-tuning (Stage 3) | Not started |
| Feedback loop (Stage 4) | Not started |

Everything left depends on one chain:

```
knowledge base  →  generator  →  ~1,650 rojak pairs  →  QLoRA  →  the chatbot
```

That chain is the deadline risk. Stages 2 and 4 are small by comparison. **Do not
spend more time on speech.** It is finished.

---

## 2. What changed since Handoff 1

1. The 9-config evaluation turned out to be **already done** — Handoff 1's
   "outstanding data-integrity task" was stale. Nothing needed re-running.
2. Three figures in the report had **drifted from the CSVs** and were corrected. The
   tables are now generated from the CSVs rather than typed in.
3. The **threshold sensitivity run was done**, with real results (§4 below).
4. Two claims in Handoff 1 turned out to be **wrong** and are corrected in §3.
5. `app/stt.py`, `app/streamlit_app.py` and a verification script were written.
6. The knowledge base was **rescoped** — narrower domain, now deviation #6.
7. The Streamlit app was **actually run** for the first time and works end to end
   (§5). `requirements.txt` now exists and pins the environment that produced it.
   The Windows environment fight it took to get there is in §12 — read it before
   setting up a second machine.

---

## 3. Two corrections to Handoff 1 — do not repeat the old claims

### The forced-English "catastrophe" is one clip

Handoff 1 §2 said forcing English on Malay audio is catastrophic, citing 6.483 WER.
Per-clip inspection says otherwise. Under forced English the eight `ms_dom` clips score:

```
0, 0, 0, 0, 1, 1, 5, 369  errors
```

Seven are fine. `s05_pq` alone enters runaway repetition and produces **362 of the 379
insertions**, against a seven-word reference. Forcing the wrong language is *usually
harmless and occasionally unbounded*.

The asymmetric threshold is still correct, but the justification is **expected cost
under a heavy tail**, not uniform damage. WER has no upper bound, so one such clip can
dominate a 48-clip corpus. Report §4 now says this; say it this way at the demo.

### `en_dom` is a routing failure, not a transcription failure

Five of the eight English-dominant clips are routed to Malay and returned as Malay
prose. The same category under forced English scores **0.106**. The model can write
these utterances correctly; the router does not ask it to.

Consequence for the "excluding the two weakest categories" line: `numeric` is a
scoring artefact from outside the system, `en_dom` is **our own component failing**.
Excluding both as equivalent flatters the result. The report now says so.

---

## 4. Threshold sensitivity — the measured answer

Run on 11 August. Full output in `eval/speech/results/detector_probs.csv` and
`threshold_sweep.csv`; written up as report §5.

**The detector is not deaf — it is under-confident.**

| Category | median `p_en` | range | routed to English |
|---|---|---|---|
| en_dom | 0.234 | 0.067 – 0.671 | 3 of 8 |
| particles | 0.011 | 0.000 – 0.466 | 0 of 8 |
| numeric | 0.002 | 0.000 – 0.524 | 1 of 8 |
| ms_dom | 0.004 | 0.000 – 0.084 | 0 of 8 |
| balanced | 0.007 | 0.000 – 0.048 | 0 of 8 |
| entity | 0.005 | 0.000 – 0.027 | 0 of 8 |

Every `en_dom` clip gets a non-trivial English probability, but the median is less
than half the 0.5 cutoff. These are English sentences carrying Malay particles by
design, so the detector hedges. **0.5 is a sensible default for monolingual audio and
the wrong scale for code-switched audio.**

The distributions **overlap**, so no cutoff separates cleanly: `en_dom` spans
0.067–0.671 while one `numeric` clip reaches 0.524 and one `particles` clip 0.466.
Routing on this signal has a floor that belongs to the detector, not the constant.

**Sweep result:** minimum 0.2765 at t = 0.05, against 0.3488 at t = 0.5 — headroom
0.072. That coincides to four decimals with the 0.277 oracle bound reached by a
different method.

### ⚠ The threshold was NOT moved, and must not be

Two reasons, and the second is the stronger one:

1. **Leakage.** Choosing a decision rule by its score on the 48 clips it is then
   evaluated on. Same objection that kept entity names out of the prompt.
2. **Instability.** At t = 0.05 one `ms_dom` clip (`s07_pq`, `p_en` = 0.084) crosses
   into English and happens to transcribe correctly either way. The dangerous clip,
   `s05_pq`, sits below and is not selected. The good result depends on *which* Malay
   clip carries the higher probability. One grid step further down, at t = 0.00, WER
   is 1.274. **The optimum sits beside a cliff.**

If asked "why 0.5?" at the demo: chosen before measurement, and the sensitivity
analysis shows what that caution costs.

### Speaker effect — report as observation, not result

Every clip routed to English belongs to **one speaker**. None of the other speaker's
24 clips crossed the threshold in any category.

| | mean `p_en` | `en_dom` median | `en_dom` max |
|---|---|---|---|
| pq | 0.136 | 0.611 | 0.671 |
| yv | 0.043 | 0.202 | 0.263 |

yv's *maximum* falls below pq's *median*. With four clips per speaker per category
this is an observation, not a demonstrated effect — but it is large, one-directional,
and it means the `en_dom` figure is partly a property of who recorded the corpus. It
is in report §8 (Threats to Validity) worded that way. The stratified test design is
what made it visible; say that too.

### Same-utterance instability — observed live, NOT yet evidence

During the first Streamlit run on 11 August, yv recorded one English sentence twice:

| take | length | `p_en` | routed | transcript |
|---|---|---|---|---|
| 1 | 4 s | 0.505 | `en` | "hello hello, can you hear me?" |
| 2 | 10 s | 0.406 | `ms` | "hello hello can you hear me" |

Same sentence, same speaker, same room, minutes apart — and opposite routing
decisions, straddling the cutoff by about ±0.05. Both transcripts are correct
English; the misroute cost punctuation, not words.

**This is deliberately not in the report.** It is n = 2, uncontrolled (the takes
differ in length), and logged nowhere, so writing it up would break the "never
hand-copy numbers into the report" rule below and invite a fair objection at the
demo. To make it reportable: record N ≥ 10 takes of one utterance through the app,
log `p_en` per take to a CSV, report the spread.

Worth doing, because it would *strengthen* report §5 rather than complicate it. That
section currently argues against tuning the threshold on two grounds — leakage, and
the cliff at t = 0.05. Test-retest spread of ±0.05 on a single utterance would be a
third and more intuitive one: the quantity being thresholded is not stable enough to
carry a precisely tuned cutoff. Note also that take 1's 0.505 exceeds every `en_dom`
clip yv recorded for the corpus (max 0.263), which hints the eval set does not span
this speaker's own range.

---

## 5. What was built this session

| File | What it is |
|---|---|
| `app/stt.py` | The shipped pipeline. `transcribe(audio) -> str`. Accepts a path, bytes, or a file object. Has a CLI. |
| `app/streamlit_app.py` | Record-and-transcribe page. **Stage 1 only** — deliberately stops at the transcript. |
| `scripts/verify_stt_matches_eval.py` | Runs `app/stt.py` over evaluation clips and diffs against the stored transcripts. |
| `eval/speech/threshold_sweep.py` | Routed WER vs threshold, from cached transcripts. No GPU. |
| `eval/speech/sync_report_figures.py` | Regenerates the report's tables from the CSVs. |

### Two guard-rails, and why they exist

`app/stt.py` **duplicates** the prompt, threshold and model names instead of importing
them — a Streamlit deployment should not need pandas and jiwer. A copy that drifts is
worse than a dependency, because the reported 0.349 would silently stop describing
what the demo runs. So:

```
python app/stt.py --check-config           # constants match the eval harness?
python scripts/verify_stt_matches_eval.py  # does it produce the same text?
```

The first compares settings, the second compares behaviour. Both have been shown to
fail when deliberately broken. **Run the second one and paste the result** — the claim
"the deployed pipeline was verified against the evaluated one" is worth making in the
report's Coding section, and it is not true until someone runs it.

### Running the Streamlit app — done 11 August, works end to end

Record in the browser, transcribe, correct rojak text out. **~6.6 s per utterance on
CPU, no GPU**, once the models are cached. First run downloads ~1.5 GB, and the CLI
reports that download inside its own timing — a 190 s first call is the download, not
the inference. The Streamlit `Time` metric is honest, because `_load_models()` runs
before the timer starts.

```
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1        # the leading .\ is required in PowerShell
pip install -r requirements.txt
winget install Gyan.FFmpeg          # reopen the terminal afterwards
python app/stt.py --check-config    # instant; all four constants matched on 11 Aug
streamlit run app/streamlit_app.py
```

**Read the header of `requirements.txt` before installing on a new machine.** Windows
long paths must be enabled first or the torch install fails in a way that blames the
wrong thing entirely — see §12. That cost about an hour.

`ffmpeg` missing is the next most likely failure.

The upload tab accepts `.mp4`, `.3gp`, `.ogg`, `.opus` and `.flac` as well as WAV/MP3/M4A,
since phone voice memos rarely arrive as `.m4a`. ffmpeg sniffs the container and ignores
the extension, so widening the whitelist was the entire change.

---

## 6. Knowledge base

**Scope decided 10 August: Malaysian government services and personal admin**, plus a
thin conversational layer. This narrows Part A §4 — it is **deviation #6**, and the
five-point justification is written in `data/knowledge_base/_scope.md`. Lead with the
first point: Part A promises a "constrained system prompt", and that instruction
cannot be written for a general scope.

| Domain | Facts | State |
|---|---|---|
| `jpj_vehicle_licence.yaml` | 18 | 13 verified against the official JPJ portal |
| `jpn_identity.yaml` | 5 | secondary sources only |
| `immigration_passport.yaml` | 5 | secondary sources only |
| `epf_socso.yaml` | 0 | not started |
| `lhdn_tax.yaml` | 0 | not started |
| `slang_meanings.yaml` | 0 | not started — Part A's *gostan* example lives here |
| `small_talk.yaml` | 0 | not started |

Plus `_out_of_scope.yaml`: **23 probes** across five categories that must trigger the
fallback, 7 marked `near_miss`.

**Two standing work items:**

1. **Upgrade the 15 `secondary` facts to `official`.** This already caught one real
   error: every aggregator reports RM120 for non-citizen driving licence renewal; the
   official JPJ table says that is the *Trial Licence* rate and CDL is RM60. Assume
   there are others.
2. **Write the medical-question response.** `_out_of_scope.yaml` flags that "saya sakit
   dada teruk" must not receive the cheerful generic fallback. It is the only probe in
   the file with real-world consequences.

**Known limitation, already written down:** narrowing shrinks but does not close the
untaught-topic hole. The base model will still answer "apa itu photosynthesis". Report
fallback rate **per category** — high for volatile/personal/current-events, lower for
out-of-domain general knowledge — rather than one flattering average.

---

## 7. Repository map

```
CLAUDE.md                          project rules
requirements.txt                   pinned app environment; READ ITS HEADER on Windows
handoffs/HANDOFF_2.md              this file — current
handoffs/HANDOFF_1.md              superseded; see §3 for what it gets wrong
app/
  stt.py                           the shipped speech pipeline
  streamlit_app.py                 Stage-1 UI
data/knowledge_base/
  README.md                        fact schema and writing rules
  _scope.md                        domain boundary + the deviation writeup
  _system_prompt.md                the constrained system prompt, draft v1
  _out_of_scope.yaml               23 fallback probes
  jpj_vehicle_licence.yaml         18 facts
  jpn_identity.yaml                5 facts
  immigration_passport.yaml        5 facts
docs/
  NLP Assignment Part A.md/.pdf    the proposal
  Rubrics.pdf, Specification.pdf   grading criteria
  PartB_Draft_STT.docx             report draft — STT sections complete
eval/speech/
  manifest.csv                     48 utterances × 2 speakers
  run_wer.py                       transcription + WER; --detect-only logs p_en
  threshold_sweep.py               WER vs threshold, no GPU
  sync_report_figures.py           report tables <- CSVs
  inspect_results.py               transcript inspection
  orthography_map.json             spelling variants for lenient scoring
  results/                         six committed result CSVs
notebooks/
  run_speech_eval.ipynb            THE Colab notebook — cells in order
scripts/
  prepare_audio.py                 phone recordings -> 16 kHz mono WAV
  verify_stt_matches_eval.py       app vs evaluation harness
```

Not yet written: `training/`, `scripts/generate_pairs.py`, `app/normalise.py`,
`app/inference.py`, `app/feedback.py`, four knowledge-base domains.

---

## 8. Workflow

```
code    lives on GitHub    ->  Colab gets it via git clone / git pull
audio   lives on Drive     ->  copied into Colab by Cell 1
results written in /content ->  Cell 3 copies them back to Drive
```

`/content` is scratch and is erased on every restart. After editing locally you **must**
`git add -A && git commit && git push`, then `!git pull` in Colab.

- GitHub: `https://github.com/yongvay/NLP-BorakBot`
- Drive audio: `MyDrive/RDS3S1/NLP/NLP Assignment/Audio`
- Drive results: `MyDrive/RDS3S1/NLP/NLP Assignment/results`
- Audio is gitignored
- **Three result locations** — `eval/speech/` (written), Drive (copied), and
  `eval/speech/results/` (committed). If they disagree, the committed ones are stale.
  This is what let three report figures drift.
- Notebook cells: 1 setup, 1b locate audio, 2 full evaluation, 3 save to Drive,
  4 inspect, 5 rescore, **6 threshold sensitivity**.

---

## 9. Deviations from Part A — state these in Part B

1. **STT moved from Week 4 to first**, on the tutor's advice.
2. **A Malaysian-fine-tuned Whisper** replaces plain `whisper-small` for transcription.
3. **Language routing** added — not in Part A, justified empirically.
4. **The §7.4 WER target was not met** (0.349 vs 0.25).
5. **Timeline compressed** — a 7-week Gantt executed in ~3 weeks.
6. **Knowledge base narrowed** from general everyday knowledge to Malaysian government
   admin plus a conversational layer. Justification in `_scope.md`.

---

## 10. Next steps, in order

1. **Write `scripts/generate_pairs.py`.** Facts in, rojak Q&A pairs out. Do this
   **before** writing more facts: if one fact does not expand into ten distinguishable
   pairs, the fix is a schema change, cheap at 28 facts and expensive at 165.
2. **Run `verify_stt_matches_eval.py`** once, and add the result to the report's Coding
   section. Five minutes.
3. **Finish the knowledge base** — 4 domains, ~137 facts. Split by agency between the
   two members, since each defends what they wrote.
4. **Generate and validate the dataset.** ~1,650 pairs, then both members review a
   sample and report agreement statistics (proposal §7.3).
5. **QLoRA round 1 on Kaggle.** LLaMA-Factory, 3B-class instruct base, 4-bit, T4.
   30 GPU-h/week, 12h session cap — checkpoint so runs survive the cut.
6. **Baseline comparison**: fine-tuned vs un-fine-tuned on perplexity, BLEU, ROUGE-L,
   BERTScore, plus human Likert. This is a second evaluation harness, not an afternoon.
7. **Normalisation (Stage 2)** and **feedback logging (Stage 4)**, then **retrain
   round 2** from the collected corrections. Round 2 is what makes the feedback loop
   demonstrable — do not let it slip.
8. **Report and demo prep.**

**If time runs short**, the things that can shrink are optional TTS, the size of the
knowledge base, and the scale of round 2. What cannot shrink: something fine-tuned,
evaluated against a baseline, and demoable.

---

## 11. Open decisions

1. **How to generate the synthetic dataset** — in-session with an assistant, a paid API
   with a reproducible script, or an open model on Kaggle/Colab. Must be reproducible
   and declared in Appendix A. *The API route is the most defensible because the script
   is the evidence.* **This blocks step 1 above.**
2. **Contribution split** for the cover page (drafted 50/50).
3. **Programme name and tutor name** for the cover page.
4. **Appendix A AI-assistance declaration** — required by TARUMT policy, must be written
   by the team, honestly and specifically.

---

## 12. Gotchas — don't repeat these

- `inspect.py` shadows Python's stdlib `inspect`, which pandas imports. Hence
  `inspect_results.py`.
- `run_wer.py --configs <subset>` used to overwrite other configs' cached transcripts.
  Now merges — but check the row count (48 × n configs).
- `prepare_audio.py` pairs raw files to manifest rows **by position**. Copying off a
  phone rewrites modification times, so use `--by-name`.
- **A validation that cannot fail is not a validation.** Both new guard-rails were
  deliberately broken to confirm they fail.
- The decoder prompt must be **code-switched**, and must share **no content with the
  test set**.
- **Never hand-copy numbers into the report.** That is exactly how three figures
  drifted. Use `sync_report_figures.py`.
- Colab drops the runtime after ~90 min idle. Re-run Cell 1 first.
- When editing a `.docx` with python-docx: paragraph indices go stale the moment you
  insert anything. Search the live list every time. Getting this wrong overwrote two
  body paragraphs with heading text.
- `_strip_prompt` had a real defect (a coarse grid search left prompt fragments in the
  transcript). It is fixed in both copies. It was **latent** — none of the 288 prompted
  transcripts took that branch — so no published figure changed. Verified before fixing.
- **Enable Windows long paths before `pip install torch`.** torch nests its third-party
  licences about seven directories deep. Under the 260-character `MAX_PATH` the install
  dies partway with `OSError: [WinError 206] The filename or extension is too long` —
  *after* `torch/` unpacks but *before* `torchgen/` does. Every later run then fails with
  `ModuleNotFoundError: No module named 'torchgen'`, which names the symptom and hides
  the cause completely. Roughly an hour lost on 11 August. Once per machine, in an
  **administrator** PowerShell, then **reboot**:
  `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`
  The repo path spends 65 characters before the venv even begins, so a venv at
  `C:\v\borak` is the no-admin alternative. Full writeup in `requirements.txt`.
- **There is a `torchgen` package on PyPI and it is not PyTorch's.** Unrelated, version
  0.0.1. Installing it to satisfy that import masks the real fault. The real `torchgen`
  ships *inside* the torch wheel — if it is missing, the wheel is broken.
- **A half-deleted package still imports.** `pip uninstall torch` refused with `no RECORD
  file`, and deleting `site-packages\torch` by hand left the DLLs behind because
  Streamlit was still running and holding them open. A directory with no `__init__.py`
  is a namespace package, so `import torch` then *succeeds*, with `__file__ = None` and
  no `__version__` — a confusing state that looks like a torch bug. Stop every
  `python.exe` first, and do **not** pass `-ErrorAction SilentlyContinue` to
  `Remove-Item`: it hides precisely the failure you need to see.
