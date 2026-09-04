# Knowledge base

The curated facts BorakBot is taught. This is a **build-time artifact**: it is
converted into rojak question-answer training pairs and baked into the model
weights by QLoRA fine-tuning. It is never loaded at inference. There is no
retrieval step (see CLAUDE.md — RAG is out of scope).

```
stage2_chatbot/knowledge_base/*.yaml   ← facts, written by hand        (this directory)
        ↓  corpus_tooling/prompts/generate_pairs_prompt.md, then build_corpus.py
stage2_chatbot/corpus/*.jsonl       ← rojak Q&A pairs, written by script
        ↓  QLoRA fine-tune
model weights
```

## Why the KB exists if nothing reads it at runtime

1. **It generates the training data.** One fact expands into ~10 paraphrased
   rojak Q&A pairs.
2. **It is the answer key for evaluation.** BLEU, ROUGE-L, BERTScore and the
   factual-consistency Likert ratings all need reference answers.
3. **It defines the fallback boundary.** Hallucination control is graded. You
   can only demonstrate that the bot declines out-of-scope questions if you know
   exactly where knowledge stops.
4. **It is the traceability story.** Rubric item 11 is individual Q&A. "Where did
   this answer come from?" has a real answer: a line in this directory, with a
   source URL and a retrieval date.

## File layout

One YAML file per domain. Filenames match the `domain` field inside.

```
_scope.md                  domain list and the in/out-of-scope boundary rule
_system_prompt.md          the inference-time domain constraint
_out_of_scope.yaml         probes that must trigger the fallback
jpj_vehicle_licence.yaml   domain files, one per agency ...
jpn_identity.yaml
immigration_passport.yaml
...
```

Files prefixed `_` are not domain files and are skipped by the generator.

One file per agency, so every fact traces to a single official portal and the
Part B results table gets clean per-domain accuracy rows.

## Fact schema

```yaml
domain: public_services
description: Malaysian government services and everyday admin.
facts:
  - id: ps.roadtax.channels
    fact: >
      Road tax can be renewed online through the MyJPJ app, the MySIKAP portal
      or MyEG, or in person at a JPJ counter, a UTC, or a Pos Malaysia branch.
    check: [MyJPJ, Pos Malaysia, JPJ]
    stability: slow
    source: https://www.pos.com.my/jpj
    retrieved: 2026-08-10
    tags: [jpj, roadtax, renewal]
```

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Unique, dotted, stable. Never reuse or renumber — training rows cite it. |
| `fact` | yes | One atomic fact, in plain English. Not rojak — register is added at generation time. |
| `check` | yes | Strings a correct answer must contain. Drives automated factual-correctness scoring (proposal §7 targets ≥85%). |
| `stability` | yes | `stable` / `slow` / `volatile`. See below. |
| `verified` | yes | `official` / `secondary` / `unverified`. See below. |
| `source` | yes | URL, or `general` for definitional facts needing no source. |
| `retrieved` | if URL | Date the URL was checked, `YYYY-MM-DD`. |
| `tags` | no | Free-form, for filtering during generation. |

### `verified`

| Value | Meaning |
|---|---|
| `official` | Confirmed against the agency's own portal, page actually opened. |
| `secondary` | Taken from a blog or aggregator. Plausible, not confirmed. |
| `unverified` | Written from memory. Should not survive to the demo. |

Every `secondary` fact is a work item: open the official page and upgrade it.
This already caught one real error — see `jpj.cdl.fee_noncitizen`, where every
aggregator reported a figure that the official JPJ table contradicts.

### `stability`

| Value | Meaning | Allowed in KB? |
|---|---|---|
| `stable` | True for years. Definitions, concepts, slang meanings. | Yes |
| `slow` | Changes every few years. Fees, rates, procedures. | Yes — but must carry a `source` and `retrieved` date |
| `volatile` | Changes daily or hourly. Prices, weather, news, exchange rates. | **No.** These belong in `_out_of_scope.yaml` and must hit the fallback. |

The `volatile` exclusion is the core of the hallucination-control story, and it
is exactly the proposal's Dialogue 3 ("Berapa harga bitcoin sekarang?").

## Rules for writing facts

1. **One fact per entry.** If a sentence contains "and" joining two independent
   claims, split it. Atomic facts paraphrase cleanly; compound ones don't.
2. **Write it in plain English.** The generator adds the rojak. Mixing registers
   here makes the facts harder to check.
3. **Every `slow` fact needs a real URL you actually opened.** Not a search
   result summary, not memory. A wrong figure surfaced at the demo is worse than
   a missing one.
4. **Prefer official sources** — `jpj.gov.my`, `kwsp.gov.my`, `jpn.gov.my`,
   `imi.gov.my`, `hasil.gov.my` — over blogs and aggregators.
5. **No personal or user-specific facts.** "How much is my EPF balance" has no
   general answer and must fall back.
6. **`check` strings must be short and literal.** `["11%"]`, not
   `["eleven percent of monthly wages"]`. They are substring-matched.
7. **If a fact might already be stale, say so in the entry** rather than
   silently trusting it.

## Adding a domain

1. Add it to the table in `_scope.md` with a one-line boundary description.
2. Create `<domain>.yaml` with `domain:`, `description:` and `facts:`.
3. Aim for 15–20 facts. Below ~15 the domain is too thin to survive the
   train/test split; above ~25 it should probably be two domains.
4. Add 2–3 near-miss probes to `_out_of_scope.yaml` — questions that *sound*
   like this domain but sit just outside it. These are what prove the boundary
   at the demo.

## Target size

~165 facts across 7 domains → ~1,650 training pairs at ~10 paraphrases each,
matching the proposal's 1,500–3,000 target. See `_scope.md` for the split.

## A note on IDs

`id` values are cited by generated training rows, so they must be stable once
generation has run. Renaming is free *before* the first
generation run and expensive after. The move from `ps.*` to
`jpj.*` / `jpn.*` / `imm.*` was made during that free window.
