# BorakBot — Handoff 3

Written 11 August 2026. **This is the current handoff.** Read it first, then
`CLAUDE.md` for the project rules. `HANDOFF_1.md` and `HANDOFF_2.md` are kept for
history; where they disagree with this one, this one is right.

---

## 1. Where the project stands

**Deadline: 31 August 2026 — 20 days.** Graded on a live demo with individual Q&A
(rubric item 11), so both members must be able to explain every design decision
in this repository.

| Module | Status |
|---|---|
| **Speech-to-text (Stage 1)** | **Closed** — built, evaluated, written up, packaged |
| **Streamlit UI shell** | **Stage 1 only** — records and transcribes, nothing else |
| **Knowledge base** | **54 facts, 5 domains** — scope narrowed, conversational layer complete |
| **Training corpus** | **632 pairs, built and validated** ← new this session |
| **Generation tooling** | **Done** — prompt, validator, corpus builder, all committed |
| QLoRA fine-tuning (Stage 3) | Not started ← **the next thing to do** |
| Rojak normalisation (Stage 2) | Not started |
| Feedback loop (Stage 4) | Not started |

The chain from Handoff 2 —

```
knowledge base  →  generator  →  pairs  →  QLoRA  →  the chatbot
```

— is now unblocked as far as QLoRA. **The critical path is now training, not
data.** Do not keep polishing the corpus; it is good enough to train on.

---

## 2. What changed since Handoff 2

1. **Scope narrowed from 7 factual domains to 3.** `epf_socso` and `lhdn_tax`
   were dropped without a fact written and their files deleted. Justified on
   verification cost, not time — full reasoning in `_scope.md`. This is
   **deviation #6 extended**.
2. **`slang_meanings` (16) and `small_talk` (10) written.** The conversational
   layer is complete. `slang.gostan` exists, so the Stage 4 demo is unblocked.
3. **Generation route decided** (Handoff 2 open decision #1, now closed) — an
   assistant chat with a committed prompt, not a Python API script. See §4.
4. **632-pair corpus built**, validated, and committed as
   `data/generated/pairs_v1.jsonl`.
5. **Two new scripts**: `validate_pairs.py` and `build_corpus.py`.
6. **Deviation #7 added** — corpus size is below the proposal's stated target.

---

## 3. One correction to Handoff 2

Handoff 2 §10 step 1 said: *"Write `scripts/generate_pairs.py`. Facts in, rojak
Q&A pairs out."* **That script was never written and should not be.** The team
chose to generate through an assistant chat rather than a paid API, so there is
no code that produces pairs. The reproducibility artifact is
`scripts/prompts/generate_pairs_prompt.md` plus the manifest recording model,
date and fact IDs per batch.

If asked at the demo "how did you generate the data?", the answer is the prompt
file and its git history — not a program. Say that plainly; a committed prompt
is a legitimate method, an invented script is not.

---

## 4. How the corpus is produced

```
data/knowledge_base/*.yaml        facts, hand-written, source-cited
        |  scripts/prompts/generate_pairs_prompt.md  (pasted into an assistant,
        |                                             5 facts per batch)
        v
data/generated/raw/*.jsonl        raw output — GITIGNORED, stays local
        |  scripts/validate_pairs.py    (gate: check strings, dupes, register)
        |  scripts/build_corpus.py      (drops padding, stamps provenance)
        v
data/generated/pairs_v1.jsonl     committed — this is what QLoRA trains on
```

**`data/generated/raw/` is gitignored** (`.gitignore:60`). Raw output stays on
the machine that produced it; only the reviewed corpus is committed. Pei Qi gets
`pairs_v1.jsonl`, not the batches.

### The two scripts

| Script | What it does |
|---|---|
| `validate_pairs.py` | Gate. Valid JSON, every `check` string present in every answer, `source_fact` exists, no duplicate questions, fallback line byte-identical, no markdown/emoji, ≤3 sentences. Exit 1 on failure. |
| `build_corpus.py` | Assembles `pairs_v1.jsonl` from raw batches. Drops cross-fact question templates and near-duplicates, stamps `batch` provenance, writes a manifest recording what was removed. |

Both have been **deliberately broken to confirm they fail** (Handoff 2's rule).
`validate_pairs.py` was tested against a corrupted check string, a hyphen
substituted for the fallback's em dash, and a duplicated question — all three
caught, exit code 1.

---

## 5. The corpus

**632 pairs. 502 answered, 130 refusal. Refusal share 20.6%. All 54 facts
covered.**

662 were generated; 30 were dropped by `build_corpus.py`:

- **16** — one question template reused across all 16 slang facts,
  `'ok kat WhatsApp orang guna "X", meaning dia apa sebenarnya?'`, with only the
  quoted word changed.
- **14** — near-duplicates, mostly a "terse" variant that was the direct
  question with punctuation stripped (`"lesen P berapa?"` / `"lesen P brape"`).

Three facts are thin after cleanup and cannot be improved: `jpj.pdl.duration`
(5), `jpj.licence.add_class` (5), `jpj.idp.fee` (7). All state a single value.

### ⚠ Known quality issue, not yet fixed

The slang answers read like dictionary entries rather than chat:

> "Alamak tu exclamation untuk dismay, surprise atau mild frustration, close to
> oh no in English."

No Malaysian says "exclamation untuk dismay". This is the English fact text with
Malay connectives inserted — the mirror image of the Bahasa Baku failure the
prompt warns against. Mean answer-similarity within a fact runs 0.60–0.71, so
the model largely restated one sentence.

**Decide before training** whether to accept this or regenerate `slang_meanings`
with a harder register instruction. It will show up in the human Likert rating
for rojak naturalness either way, so it is better to have decided deliberately
than to be asked about it.

---

## 6. Knowledge base

**54 facts across 5 domains. 39 `official`, 15 `secondary`.**

| Domain | Facts | Target | State |
|---|---|---|---|
| `jpj_vehicle_licence` | 18 | 18 ✔ | All fees official; 5 road-tax procedure facts from an aggregator |
| `slang_meanings` | 16 | 16 ✔ | Definitional, `source: general` |
| `small_talk` | 10 | 10 ✔ | Definitional, `source: general` |
| `jpn_identity` | 5 | ~15 | **All five are about a lost MyKad** |
| `immigration_passport` | 5 | ~15 | **All five are about renewal** |

JPN and Imigresen are **narrow as well as thin**. Someone applying for a first
MyKad or a child's passport hits the fallback on a question squarely inside the
stated domain — which looks like the boundary failing rather than working. Fix
by breadth, not depth: topic lists are in `_scope.md`.

Four `small_talk` facts carry `check: []` **deliberately** — they are
performative. The correct reply to "hi" is a greeting, not a sentence containing
the word "greeting", so there is no factual claim to substring-match. Judge them
by the Likert ratings. Expect to be asked about this; the reasoning is in the
file header.

---

## 7. Findings worth reporting in Part B

These came out of building the corpus and are more interesting than the corpus.

1. **`check` strings are matched against a *rojak* answer, case-sensitively.**
   An English gloss fails: `talk.not_human` originally carried
   `check: [not a human]`, but the natural reply is "Saya bukan manusia lah".
   Nine of the first fourteen validator failures were a check string not
   matching a capitalised sentence-initial word (`"road tax"` vs `"Road tax"`).
   **Decide and document** whether matching is case-sensitive — it silently
   moves the §7 factual-correctness figure either way.

2. **Single-value facts do not yield 10 paraphrases.** Facts stating one number
   support ~5–7 genuinely distinct questions; multi-value facts support 10. The
   proposal's *165 facts × 10 = 1,650* arithmetic assumed a uniform yield that
   does not exist.

3. **Generators pad by reusing a question frame across facts**, which an
   exact-match duplicate check cannot see — 16 distinct strings, one question.
   Detecting it needs skeletonisation (strip the quoted term, compare shapes).
   Both generators used on this project padded; 14 of the 30 dropped pairs came
   from the earlier batches, not the later ones.

4. **Performative facts need an empty check list.** A conversational reply
   cannot contain a description of itself.

---

## 8. Deviations from Part A — state these in Part B

1. **STT moved from Week 4 to first**, on the tutor's advice.
2. **A Malaysian-fine-tuned Whisper** replaces plain `whisper-small`.
3. **Language routing** added — not in Part A, justified empirically.
4. **The §7.4 WER target was not met** (0.349 vs 0.25).
5. **Timeline compressed** — a 7-week Gantt executed in ~3 weeks.
6. **Knowledge base narrowed twice** — from general everyday knowledge to
   Malaysian admin (10 Aug), then from five factual domains to three (11 Aug).
   Justification in `_scope.md`; lead with verifiability, not time.
7. **Training set below the proposal target** — 632 pairs against a stated
   1,500–3,000. The trade was breadth for verifiability, and the whole pipeline
   is reproducible from committed artifacts.

---

## 9. Next steps, in order

1. **Fix `_system_prompt.md`.** It still names "KWSP/EPF, SOCSO, dan cukai
   pendapatan LHDN" as in-domain. The bot would advertise knowledge it does not
   have and be provoked into inventing it — the exact failure being measured.
   **Blocking. Ten minutes.**
2. **Decide the train/val/test split** — see §10, it is not the obvious choice.
3. **Convert to LLaMA-Factory format** (sharegpt or alpaca) and write
   `training/qlora_config.yaml`.
4. **QLoRA round 1 on Kaggle.** 3B-class instruct base, 4-bit, T4. 30 GPU-h/week,
   12h session cap — checkpoint so runs survive the cut.
5. **Baseline comparison**: fine-tuned vs un-fine-tuned on perplexity, BLEU,
   ROUGE-L, BERTScore, plus human Likert. A second evaluation harness, not an
   afternoon.
6. **Upgrade the 15 `secondary` facts.** More urgent than before: ten pairs were
   generated from each, so any error is multiplied by ten in the corpus.
7. **Broaden JPN and Imigresen** (+20 facts), then regenerate those domains.
8. **Write the medical-question response** for "saya sakit dada teruk".
9. **Normalisation (Stage 2)** and **feedback logging (Stage 4)**, then
   **retrain round 2** from collected corrections. Round 2 is what makes the
   feedback loop demonstrable — do not let it slip.
10. **Report and demo prep.**

**If time runs short**, what can shrink: optional TTS, the JPN/Imigresen
broadening, the scale of round 2. What cannot: something fine-tuned, evaluated
against a baseline, and demoable.

---

## 10. Open decisions

1. **Train/val/test split — pair-level or fact-level?** This is not obvious and
   is worth getting right.

   *Fact-level* (hold out whole facts) measures generalisation to unseen facts —
   but the knowledge is baked into the weights by design, with no retrieval
   step, so the model **cannot** answer a fact it was never trained on. That
   test is guaranteed to fail and measures nothing useful.

   *Pair-level* (hold out some paraphrases of every fact) measures whether the
   model learned the fact robustly across phrasings rather than memorising one
   surface form. That is the thing this architecture can actually be good at.

   **Recommendation: pair-level, reported honestly** as paraphrase robustness,
   not generalisation to unseen knowledge. Say so before a tutor asks whether
   the split leaks — it does, deliberately, and the justification is that
   memorisation is the intended mechanism.

2. **Whether to regenerate `slang_meanings`** for register — see §5.
3. **Case sensitivity of `check` matching** — see §7.1.
4. **Contribution split** for the cover page (drafted 50/50).
5. **Programme name and tutor name** for the cover page.
6. **Appendix A AI-assistance declaration.** Must be written by the team,
   honestly and specifically. It now needs to cover: training pairs generated by
   an assistant from a hand-written knowledge base, per-batch model and date in
   `pairs_v1_manifest.json`, and the fact that the facts themselves are the
   team's own work.

---

## 11. Repository map — changes since Handoff 2

```
data/knowledge_base/
  _scope.md                        REWRITTEN — 5 domains, deviation #7, cascade notes
  slang_meanings.yaml              NEW — 16 facts
  small_talk.yaml                  NEW — 10 facts
  epf_socso.yaml                   DELETED
  lhdn_tax.yaml                    DELETED
  _system_prompt.md                STALE — still names dropped domains
data/generated/
  pairs_v1.jsonl                   NEW — 632 pairs, committed
  pairs_v1_manifest.json           NEW — provenance and what was dropped
  raw/                             gitignored, local only
scripts/
  validate_pairs.py                NEW — corpus gate
  build_corpus.py                  NEW — raw -> reviewed corpus
  prompts/generate_pairs_prompt.md NEW — the generation method, and the evidence
```

Not yet written: `training/`, `app/normalise.py`, `app/inference.py`,
`app/feedback.py`.

---

## 12. Gotchas — new ones, in addition to Handoff 2 §12

- **`check` strings are case-sensitive and matched against rojak.** See §7.1.
  This cost fourteen rewrites on the first validator run.
- **Single-value facts cap out around 5–7 pairs.** Do not pad; the prompt now
  tells the model to return fewer real questions instead.
- **An exact-match duplicate check does not catch a reused question template.**
  Skeletonise before comparing.
- **A concatenated "all pairs" file will double-count.** ChatGPT returned
  `all_training_pairs.jsonl` alongside the six batches; `build_corpus.py`
  excludes it by name.
- **Raw generation output was once committed to the repo root** and to
  `borakbot_training_pairs_raw/`. Both were removed. Check `git status` before
  committing after unzipping anything into the repo.
- **`data/generated/raw/` is gitignored**, so `git add` silently skips it. If a
  file seems not to commit, run `git check-ignore -v <path>`.
