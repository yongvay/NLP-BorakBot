# Knowledge base scope

Decided 10 August 2026. **This narrows Part A §4 and must be declared as a
deviation in Part B** — see "Deviation from Part A" below.

---

## The domain

> **Malaysian everyday government services and personal admin** — road tax,
> driving licences, MyKad, passports, EPF/SOCSO, and income tax basics.
>
> Plus a thin conversational layer: greetings, bot identity, and Manglish slang
> meanings.

Factual knowledge is narrow. Conversational **register** stays broad — the bot
still talks rojak about anything, it just only *knows facts* about admin.

---

## Deviation from Part A

Part A §4 promised "general conversational domains (everyday knowledge Q&A,
daily-life advice, public-service and how-to enquiries, small talk,
campus/student life)". This scope keeps public-service enquiries, small talk and
slang, and drops general knowledge, daily-life advice and campus life.

### Justification (use this wording in Part B)

1. **The constrained system prompt becomes implementable.** Part A §6.1 Stage 3
   specifies a "constrained system prompt" that keeps the model inside its
   trained knowledge. That instruction cannot be written for a general scope —
   "only answer general everyday questions" constrains nothing. "Only answer
   questions about Malaysian government services and personal admin" is a
   constraint the model can follow, and it operates at inference time on top of
   the training-time fix. The safeguard promised in the proposal only exists if
   the domain is nameable.

2. **Hallucination control is graded and becomes demonstrable.** With a bounded
   domain, out-of-scope questions are structurally different from in-scope ones,
   so refusal generalises rather than being memorised case by case.

3. **Factual correctness becomes measurable.** The ≥85% factual correctness rate
   in Part A §7 requires an enumerable set of in-knowledge answers.

4. **Fine-tuning budget is finite.** ~1,500 pairs of QLoRA on a 3B model is a
   small budget. Spread across nine domains it teaches register but not
   knowledge, and a model that has learned the style without the facts fills
   gaps by bluffing — the exact failure the project is meant to control.

5. **It matches the STT evaluation already completed.** The `entity` category in
   the speech test set is built from Malaysian admin entities (MyJPJ, JPJ).
   Narrowing aligns the two halves of the system.

The project's contribution is Malay-English code-switched dialogue, not breadth
of facts. Narrowing the knowledge does not narrow the contribution.

---

## The boundary

### In scope — all three must hold

| Property | Test |
|---|---|
| **In domain** | About Malaysian government services or personal admin. |
| **Stable** | Still true next month. Procedures and fees, not live values. |
| **General** | Same answer for everyone who asks. |

### Out of scope — any one is enough

| Category | Example | Why |
|---|---|---|
| **Out of domain** | "Apa itu photosynthesis?" | Not admin. Now cleanly outside, which it was not under the general scope. |
| **Volatile** | "Berapa harga bitcoin sekarang?" | Frozen weights cannot know live values. Part A Dialogue 3. |
| **Personal** | "Berapa EPF balance saya?" | Needs account access the bot does not have. |
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

### Factual core — Malaysian admin

| # | Domain | File | Agency | Target |
|---|---|---|---|---|
| 1 | Vehicle and driving licence | `jpj_vehicle_licence.yaml` | JPJ | ~35 |
| 2 | Identity documents | `jpn_identity.yaml` | JPN | ~20 |
| 3 | Passport and travel | `immigration_passport.yaml` | Imigresen | ~20 |
| 4 | Retirement and social security | `epf_socso.yaml` | KWSP, PERKESO | ~25 |
| 5 | Income tax | `lhdn_tax.yaml` | LHDN | ~25 |

One file per agency, so every fact traces to one official portal. This also
gives clean per-domain accuracy rows in the Part B results table, and it is the
natural line along which to split work between the two team members.

### Conversational layer

| # | Domain | File | Target |
|---|---|---|---|
| 6 | Slang meanings | `slang_meanings.yaml` | ~25 |
| 7 | Small talk | `small_talk.yaml` | ~15 |

`slang_meanings` does double duty — it teaches vocabulary and demonstrates rojak
competence, and it is where Part A's feedback-loop example (*gostan*) lives, so
it must exist before the Stage 4 demo.

`small_talk` includes bot identity ("who are you", "what can you do"). The
"what can you do" answer should state the domain out loud, so the bot advertises
its own boundary instead of leaving users to discover it by being refused.

**Total target: ~165 facts → ~1,650 training pairs at ~10 paraphrases each.**

---

## Status

| Domain | Facts | `official` | `secondary` | State |
|---|---|---|---|---|
| jpj_vehicle_licence | 18 | 13 | 5 | Fees verified against the official JPJ table |
| jpn_identity | 5 | 0 | 5 | Needs verification against jpn.gov.my |
| immigration_passport | 5 | 0 | 5 | Needs verification against imi.gov.my |
| epf_socso | 0 | — | — | Not started |
| lhdn_tax | 0 | — | — | Not started |
| slang_meanings | 0 | — | — | Not started |
| small_talk | 0 | — | — | Not started |

**28 of ~165 facts. 13 `official`, 15 `secondary`.**

Out-of-scope probes: 23 across 5 categories, 7 marked `near_miss`.

Two standing work items:

1. Upgrade all 15 `secondary` facts to `official` before the demo. One error has
   already been caught this way (`jpj.cdl.fee_noncitizen`), so assume there are
   others.
2. Write the medical-question response template flagged in `_out_of_scope.yaml`.
   It must not receive the generic fallback.
