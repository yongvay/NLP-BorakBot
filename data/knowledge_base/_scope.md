# Knowledge base scope

Decided 10 August 2026, **narrowed again 11 August 2026**. This narrows Part A §4
and must be declared as a deviation in Part B — see "Deviation from Part A"
below. The 11 August change also produces a second, separate deviation about
training-set size.

---

## The domain

> **Malaysian identity, travel and vehicle admin** — road tax, driving licences,
> MyKad and passports.
>
> Plus a conversational layer: greetings, bot identity, and Manglish slang
> meanings.

Factual knowledge is narrow. Conversational **register** stays broad — the bot
still talks rojak about anything, it just only *knows facts* about these three
agencies.

---

## What changed on 11 August, and why

The 10 August scope named five factual domains. Two of them — `epf_socso`
(KWSP/PERKESO) and `lhdn_tax` (LHDN) — were **dropped without a single fact
written**, and their files deleted.

The reason is verification cost, not topic. Every fact in this knowledge base
needs a figure confirmed against the agency's own portal, and the two dropped
domains carry the highest risk of a wrong one:

- **EPF restructured its accounts in 2024.** The old Account 1 / Account 2 split
  is gone. Nearly every secondary source still describes the old structure, and
  so will the base model.
- **Tax reliefs and bands change with each Budget**, so a correct figure has a
  short shelf life and must carry an assessment year.
- **LHDN sits directly against a fallback category.** "How does e-Filing work"
  is in scope; "how should I declare to pay less tax" is professional advice.
  Facts near that line are the easiest to write wrongly.

With 20 days to the deadline and QLoRA, a baseline comparison harness,
normalisation, the feedback loop and the report still outstanding, ~50 facts of
high-risk verification was the wrong place to spend the remaining time. Three
domains fully sourced beats five domains half-sourced, because the failure mode
of a half-sourced domain is a confident wrong fee on screen at the demo.

**State it that way in Part B.** The reduction was a verifiability decision, and
it is defensible on those grounds; presenting it as running out of time is not.

---

## Deviation from Part A

Part A §4 promised "general conversational domains (everyday knowledge Q&A,
daily-life advice, public-service and how-to enquiries, small talk,
campus/student life)". This scope keeps a subset of public-service enquiries,
small talk and slang, and drops general knowledge, daily-life advice, campus
life, and — as of 11 August — retirement, social security and income tax.

### Justification (use this wording in Part B)

1. **The constrained system prompt becomes implementable.** Part A §6.1 Stage 3
   specifies a "constrained system prompt" that keeps the model inside its
   trained knowledge. That instruction cannot be written for a general scope —
   "only answer general everyday questions" constrains nothing. "Only answer
   questions about road tax, driving licences, MyKad and passports" is a
   constraint the model can follow, and it operates at inference time on top of
   the training-time fix. The safeguard promised in the proposal only exists if
   the domain is nameable.

2. **Hallucination control is graded and becomes demonstrable.** With a bounded
   domain, out-of-scope questions are structurally different from in-scope ones,
   so refusal generalises rather than being memorised case by case.

3. **Factual correctness becomes measurable.** The ≥85% factual correctness rate
   in Part A §7 requires an enumerable set of in-knowledge answers.

4. **Fine-tuning budget is finite.** ~630 pairs of QLoRA on a 3B model is a small
   budget. Spread across nine domains it teaches register but not knowledge, and
   a model that has learned the style without the facts fills gaps by bluffing —
   the exact failure the project is meant to control.

5. **It matches the STT evaluation already completed.** The `entity` category in
   the speech test set is built from Malaysian admin entities (MyJPJ, JPJ).
   Narrowing aligns the two halves of the system.

6. **Every remaining fact can be traced to one agency portal.** Three agencies,
   three files, three source domains. This is what makes rubric item 11 —
   "where did this answer come from?" — answerable with a line number.

The project's contribution is Malay-English code-switched dialogue, not breadth
of facts. Narrowing the knowledge does not narrow the contribution.

---

## The boundary

### In scope — all three must hold

| Property | Test |
|---|---|
| **In domain** | About JPJ vehicle/licence matters, JPN identity documents, or Malaysian passports. Or a slang or conversational question. |
| **Stable** | Still true next month. Procedures and fees, not live values. |
| **General** | Same answer for everyone who asks. |

### Out of scope — any one is enough

| Category | Example | Why |
|---|---|---|
| **Out of domain** | "Apa itu photosynthesis?", "Berapa caruman EPF saya?" | Not one of the three agencies. EPF, SOCSO and income tax moved into this row on 11 August. |
| **Volatile** | "Berapa harga bitcoin sekarang?" | Frozen weights cannot know live values. Part A Dialogue 3. |
| **Personal** | "Bila roadtax saya expired?" | Needs account access the bot does not have. |
| **Current events** | "Siapa menang bola semalam?" | Post-training by construction. |
| **Professional advice** | "Boleh saman company saya tak?" | Legal, medical and specific financial advice need a professional. |

### ⚠ Remaining limitation — still report this

Narrowing shrinks the untaught-topic hole but does not close it. The base model
retains its pre-training knowledge, and a 3B instruct model asked about
photosynthesis may still answer despite the system prompt and the training-time
refusals. Suppression is imperfect.

Report fallback rate **per category** in Part B. Expect it to be high for
volatile, personal and current-events probes (strong surface cues, explicitly
trained) and lower for out-of-domain general knowledge (competing against the
base model's own priors). Stating this before the tutor finds it is worth more
than a clean-looking single number.

---

## Domains

### Factual core

| # | Domain | File | Agency | Facts | Target |
|---|---|---|---|---|---|
| 1 | Vehicle and driving licence | `jpj_vehicle_licence.yaml` | JPJ | 18 | 18 ✔ |
| 2 | Identity documents | `jpn_identity.yaml` | JPN | 5 | ~15 |
| 3 | Passport and travel | `immigration_passport.yaml` | Imigresen | 5 | ~15 |

One file per agency, so every fact traces to one official portal. This also
gives clean per-domain accuracy rows in the Part B results table, and it is the
natural line along which to split work between the two team members.

### Conversational layer

| # | Domain | File | Facts | Target |
|---|---|---|---|---|
| 4 | Slang meanings | `slang_meanings.yaml` | 16 | 16 ✔ |
| 5 | Small talk | `small_talk.yaml` | 10 | 10 ✔ |

`slang_meanings` does double duty — it teaches vocabulary and demonstrates rojak
competence, and it is where Part A's feedback-loop example (*gostan*) lives, so
it must exist before the Stage 4 demo. `slang.terror` and `slang.action` are
false friends against standard English and are the strongest fine-tuned-vs-base
comparison items in the file.

`small_talk` includes bot identity. The `talk.capability` fact states the domain
out loud, so the bot advertises its own boundary instead of leaving users to
discover it by being refused. **`talk.capability` and this document must agree** —
if one changes, change the other.

Four `small_talk` facts carry `check: []` deliberately. They are performative:
the correct reply to "hi" is a greeting, not a sentence describing what a
greeting is, so there is no factual claim to substring-match. Judge those by the
human Likert ratings for fluency and rojak naturalness instead.

**Current total: 54 facts → 632 training pairs. Target: ~74 facts → ~700 pairs.**

---

## Deviation #7 — training set below the proposal target

Part A §7 commits to 1,500–3,000 training pairs. The corpus is **632** and will
reach roughly 700 when JPN and Imigresen are filled out. This is a second,
separate deviation and must be declared alongside the scope narrowing.

The trade was breadth for verifiability: every fee in the corpus traces to an
agency portal, and the generation prompt, validator and corpus-assembly script
are committed so the whole set is reproducible. That is a better position to
defend than a larger corpus assembled from sources the team cannot vouch for.

Report the pair count honestly, next to the fallback-rate-per-category table.

---

## Consequences of the 11 August narrowing — do not miss these

Dropping EPF and LHDN changes three things elsewhere in the repository.

1. **`_system_prompt.md` draft v1 is now WRONG.** It still names "KWSP/EPF,
   SOCSO, dan cukai pendapatan LHDN" as in-domain. Left unfixed, the bot
   advertises knowledge it does not have and will be provoked into inventing
   it — the exact failure this project measures. **Fix before any fine-tuning
   run.**

2. **Three near-miss probes have been downgraded.** In `_out_of_scope.yaml`,
   `Berapa EPF balance saya sekarang?`, `cukai saya kena bayar berapa tahun ni?`
   and `saya declare income tax macam ni boleh elak cukai tak?` were marked
   `near_miss: true` because their *topic* was in scope and only the personal or
   advisory framing pushed them out. Now the topic is out of scope too, so they
   are ordinary `out_of_domain` refusals and no longer demonstrate a subtle
   boundary. Either re-mark them or stop citing them as near-misses.

   Four genuine near-misses survive, all JPJ/JPN/Imigresen:
   `my roadtax expired ke belum ah?`, `saya ada saman JPJ tak?`,
   `passport saya dah siap ke belum?`, and
   `JPJ ada buat apa-apa announcement baru tak minggu ni?`
   The first is still the best demo pair in the file — ask
   "macam mana nak renew roadtax?" first, then it.

3. **`README.md` uses `ps.roadtax.channels` in its schema example**, an ID
   prefix that no longer exists. Cosmetic, but it is the first thing a reader
   sees.

---

## Status

| Domain | Facts | `official` | `secondary` | State |
|---|---|---|---|---|
| jpj_vehicle_licence | 18 | 13 | 5 | All fees verified; the 5 road-tax procedure facts came from an aggregator |
| jpn_identity | 5 | 0 | 5 | All five are about a lost MyKad — narrow as well as thin |
| immigration_passport | 5 | 0 | 5 | All five are about renewal — no first-time, child or lost-passport facts |
| slang_meanings | 16 | 16 | 0 | Definitional, `source: general`, no portal needed |
| small_talk | 10 | 10 | 0 | Definitional, `source: general`, no portal needed |

**54 facts. 39 `official`, 15 `secondary`.**

Out-of-scope probes: 23 across 5 categories, 7 marked `near_miss` — see
"Consequences" above, three of those seven no longer earn the label.

### Standing work items

1. **Upgrade all 15 `secondary` facts to `official`.** This already caught one
   real error (`jpj.cdl.fee_noncitizen`: every aggregator reports RM120, the
   official JPJ table says that is the Trial Licence rate and CDL is RM60), so
   assume there are others. This is now more urgent than it was: ten training
   pairs were generated from each of those facts, so any error is already
   multiplied by ten in the corpus.

2. **Broaden JPN and Imigresen, not just deepen them.** Both currently answer
   one question each. Someone applying for a first MyKad or a child's passport
   gets the fallback on a question squarely inside the stated domain — which
   looks like the boundary failing rather than working.

3. **Write the medical-question response.** `_out_of_scope.yaml` flags that
   "saya sakit dada teruk" must not receive the cheerful generic fallback. It is
   the only probe in the file with real-world consequences, and the one a tutor
   is most likely to try.

4. **Fix `_system_prompt.md`** — see Consequences #1.
