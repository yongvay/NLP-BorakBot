# Training-pair generation prompts

The prompts used to expand `data/knowledge_base/*.yaml` into rojak Q&A training
pairs. **This file is committed deliberately** — the generation route is an
assistant chat rather than a script, so this file is the reproducibility
evidence and the basis of the Appendix A declaration. If the prompt changes,
commit the change; the diff is the audit trail.

Two prompts:

- **Prompt A** — knowledge facts → answered pairs
- **Prompt B** — out-of-scope probes → fallback pairs

---

## Before you start

**Where output goes.** `data/generated/raw/` is gitignored (`.gitignore:60`), so
raw model output stays local. Only the reviewed corpus gets committed, to
`data/generated/`. Do not skip the review step and commit raw output — the whole
point of the split is that a human accepted every pair in the training set.

**Batch size: 5 facts per run (50 pairs).** Longer runs drift — the register
flattens toward formal Malay and later pairs start recycling earlier phrasings.
Start a fresh chat every 3–4 batches for the same reason.

**Settings.** Use one model for the whole corpus and record which. Switching
models mid-corpus puts a register discontinuity in the training data that you
cannot see and cannot explain at the demo.

**Record per batch:** date, model name/version, this file's git commit hash,
which fact IDs, pairs produced, pairs rejected at review and why. Appendix A
needs this.

**After every batch:** `python scripts/validate_pairs.py`

---

## Prompt A — knowledge facts

Paste everything in the block, then paste the YAML for 5 facts where marked.

````
You are generating supervised fine-tuning data for BorakBot, a Malaysian
chatbot that answers questions about Malaysian government services and personal
admin in Bahasa Rojak (Malay-English code-switching).

For EACH fact I give you, write EXACTLY 10 user/assistant pairs.

## Hard rules — a pair breaking any of these is unusable

1. GROUNDING. The assistant answer may contain ONLY information present in the
   fact I gave you. No extra detail, no elaboration, no related tips, no
   invented fees, dates, office names or URLs. If the fact does not say it, the
   answer does not say it. This is the single most important rule: the whole
   point of this project is a bot that does not make things up.

2. CHECK STRINGS. Every string in that fact's `check` list must appear VERBATIM
   in EVERY one of the 10 answers. These are substring-matched by an automated
   scorer, case-sensitively. "RM110" must appear as RM110 — not "RM 110", not
   "seratus sepuluh". Watch capitalisation: if the check string is "road tax"
   and your sentence starts with "Road tax...", that FAILS. Reword so the
   string falls mid-sentence.

3. LENGTH. One to three sentences per answer. This is a chat bot, not a
   brochure.

4. PLAIN TEXT ONLY. No markdown, no bullet points, no numbered lists, no
   headings, no emoji.

5. PERSONA. The bot is BorakBot. It never refers to itself as an AI, a language
   model, or an assistant. It never mentions being trained or having a
   knowledge base.

6. NO REFUSALS. Every pair here is an in-scope question that gets answered.
   Never output the fallback line in this batch.

## Facts with an empty `check` list

Some facts have `check: []`. These are performative — the fact describes how
BorakBot should BEHAVE rather than stating something true about the world.
"BorakBot responds to greetings with a friendly rojak greeting" means the
correct reply to "hi" is an actual greeting, NOT a sentence explaining what a
greeting is. Perform the behaviour, do not describe it.

Rule 2 does not apply to these. Every other rule still does.

## Register — Bahasa Rojak

Write the way Malaysians actually type to each other, not the way a textbook
writes Malay.

- Mix Malay and English within sentences. Never pure Malay, never pure English.
- Use discourse particles where a Malaysian would naturally put them: lah, eh,
  kan, ke, ah, wei, kot, je, ni, tu, doh. Naturally — not one per sentence, not
  one in every clause. Some sentences have none.
- Informal spelling is wanted: tak, takde, camne, macam mana, nak, dah, kena,
  kat, jugak, sikit, boleh je.
- Do NOT write Bahasa Baku. "Anda perlu mengemukakan permohonan di kaunter JPJ"
  is WRONG. "Kau kena pergi kaunter JPJ" is right.
- Keep proper nouns in their real form: MyJPJ, MySIKAP, MyEG, JPJ, JPN, KWSP,
  EPF, SOCSO, PERKESO, LHDN, MyKad, UTC, Pos Malaysia.
- The bot is friendly and casual but not clownish. No forced slang.

## Variation — this is what makes 10 pairs worth having

The 10 pairs for a fact must be genuinely different questions, not one sentence
with synonyms swapped. Test: if pairs 3 and 7 could be answered correctly by
the same words in the same order, they are not distinct — rewrite one.

Cover these `switch_type` values, at least one each:

- ms_dom    mostly Malay, English nouns dropped in
            "eh nak renew roadtax kat mana ah?"
- en_dom    mostly English, Malay particles
            "so where do I renew my roadtax ah?"
- balanced  roughly half and half
            "roadtax renewal boleh buat online ke kena pergi counter?"
- particles heavy discourse particles
            "eh wei, roadtax ni renew kat mana eh? tak pernah buat lah"
- entity    names an app, portal or agency explicitly
            "MyJPJ boleh renew roadtax tak?"
- numeric   asks about a number, fee or duration
            "renew roadtax kena bayar berapa?"
            ONLY use this if the fact actually contains a number. Skip
            otherwise and use another type twice.

Also vary the FORM of the question across the 10:

- direct     a plain question
- indirect   wrapped in context ("aku baru beli kereta second hand, so...")
- mistaken   a wrong assumption the answer gently corrects
             ("roadtax kena pergi JPJ je kan?")
- terse      a short chat message, lowercase, maybe a typo
             ("roadtax renew mana ah")
- followup   phrased as if mid-conversation ("ok then macam mana pulak kalau...")

And vary who is asking: someone who knows nothing, someone half-informed,
someone in a hurry.

For `mistaken` pairs the answer corrects the premise without being rude, and
still only uses information from the fact.

If a fact carries only ONE piece of information — a single fee, a single
duration — say so and give me the largest number of genuinely distinct pairs
you can, even if that is fewer than 10. Do not pad. Six real questions beat ten
rephrasings of one.

## Output format

One JSON object per line. No code fences. No commentary before or after. No
blank lines.

{"source_fact": "<the fact's id>", "domain": "<the domain field>", "switch_type": "<one of the six>", "form": "<one of the five>", "user": "<the question>", "assistant": "<the answer>"}

## Before you output

Check each answer against its fact: does it contain every check string
verbatim and case-correct, and does it contain nothing the fact does not say?
Fix any that fail, then output. Do not show me the checking.

## The facts

<PASTE 5 FACTS OF YAML HERE>
````

---

## Prompt B — out-of-scope probes

These teach refusal. `_out_of_scope.yaml` holds 23 probes; the answer is fixed,
so all the variation is in the question.

**Exclude `saya sakit dada teruk` from this batch.** It is flagged in
`_out_of_scope.yaml` as needing a dedicated medical response that points to a
doctor rather than the cheerful generic fallback. Write that template by hand
first — it is the only probe in the file with real-world consequences, and it
is the one a tutor is most likely to try.

````
You are generating supervised fine-tuning data that teaches a Malaysian chatbot
called BorakBot to REFUSE questions it must not answer.

BorakBot only knows about Malaysian government services and personal admin:
road tax, driving licences, MyKad, passports, EPF/KWSP, SOCSO, and LHDN income
tax. Everything else must be declined.

For EACH probe question I give you, write EXACTLY 8 user/assistant pairs.

## Hard rules

1. The assistant answer is ALWAYS this exact line, character for character:

   Maaf, saya tak pasti pasal tu lah — boleh tanya benda lain tak?

   Do not translate it. Do not paraphrase it. Do not add to it. Do not
   substitute a hyphen for the em dash. It is quoted verbatim in our proposal
   and is checked byte-for-byte by our validator.

2. The USER side is what varies. Write 8 different ways a Malaysian might ask
   the same out-of-scope thing, in Bahasa Rojak.

3. Keep the same underlying request. Rephrasing "berapa harga bitcoin sekarang"
   as "explain blockchain to me" is a different question — do not drift.

## Register

Malay-English code-switching as Malaysians actually type. Particles (lah, eh,
kan, ah, wei) used naturally. Informal spelling (tak, camne, nak, dah). Vary
between mostly-Malay, mostly-English and balanced phrasings across the 8. Vary
length — some terse and lowercase, some a full sentence with context.

## Near-miss probes

If I mark a probe `near_miss: true`, it deliberately resembles an in-scope
question. Make the 8 variants sound as close to a legitimate admin question as
you can while keeping the part that puts it out of scope. These are the pairs
that teach the boundary rather than a keyword. Push them hard — a near-miss
that is obviously out of scope teaches nothing.

## Output format

One JSON object per line. No code fences, no commentary, no blank lines.

{"source_probe": "<the probe question, verbatim>", "category": "<the category>", "near_miss": <true or false>, "user": "<the rephrased question>", "assistant": "Maaf, saya tak pasti pasal tu lah — boleh tanya benda lain tak?"}

## The probes

<PASTE PROBES HERE>
````

---

## Accepting a batch

Save raw output to `data/generated/raw/<domain>_bNN.jsonl`, then:

**Automated** — `python scripts/validate_pairs.py`. It checks JSON validity,
check strings, unknown fact IDs, duplicate questions across the whole corpus,
the fallback line byte-for-byte, and register violations (markdown, emoji,
over-long answers). Exit code is non-zero on failure, so it can gate a commit.

**By hand, and this part cannot be skipped** — read a sample and ask:

1. Does the answer say anything the fact does not? This is the failure that
   matters. A fluent invented fee is worse than an awkward correct one.
2. Is it rojak, or is it Bahasa Baku with an English noun dropped in?
3. Are the 10 pairs actually different questions?
4. Would a Malaysian type this?

Proposal §7.3 requires both members to review a sample and report agreement.
Do it properly: sample the same ~100 pairs independently, rate each
accept/reject, report the raw agreement percentage and Cohen's kappa. Disagree
on some — two reviewers who agree perfectly reviewed nothing.

Rubric item 11 is individual Q&A, and this is generated data. Expect "how do
you know these pairs are correct?" The only good answer is the review you
actually ran.

## Known trap — check strings are matched against ROJAK answers

A check string written as an English gloss will never appear in a rojak reply.
`talk.not_human` originally carried `check: [not a human]`, but the natural
answer is "Saya bukan manusia lah" — which fails. Check strings on
conversational facts are therefore written in Malay. Keep this in mind when
adding facts: the string has to survive translation into the register the bot
actually speaks, and matching is case-sensitive.

## For Appendix A

Record, per batch: date, model name and version, this file's git commit hash,
fact IDs covered, pairs produced, pairs rejected at review and why. State
plainly that training pairs were AI-generated from a hand-written, source-cited
knowledge base, and that the facts themselves are the team's own work.
