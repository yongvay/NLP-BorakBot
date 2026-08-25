# BorakBot — design notes

Every non-obvious decision, with the evidence behind it. The code carries one-line comments
pointing at the sections here. This is the document to read before the demo.

- **§1–§5 — Stage 1, speech-to-text.** Decisions in `app/stt.py`.
- **§6–§7 — Stage 3, fine-tuning.** Training environment and base-model choice.

---

# Stage 1 decisions (speech-to-text)

All numbers come from `archive/eval/speech/results/`, produced by
`archive/eval/speech/run_wer.py` over 48 clips / 387 reference words, two speakers, six
code-switching categories.

## §1 Why two models instead of one

`mesolitica/malaysian-whisper-small-v3` transcribes Malay and rojak far better than base
Whisper, but its own language detector always answers Malay — Mesolitica pins the Malay
token in `generation_config`. On English-dominant clips it therefore *translates* instead of
transcribing.

Base `whisper-small`'s detector is unbiased and costs one encoder pass (~0.3 s). So base
Whisper picks the language token and the Malaysian model does all the transcribing, always.
Base Whisper never produces a word of the output.

That this actually matters is visible in the config sweep — the only difference between these
two rows is which language token the transcriber was handed:

| Configuration | WER (strict) |
|---|---|
| `malaysian_prompt_routed` — routed by base Whisper | **0.3488** |
| `malaysian_prompt_auto` — Mesolitica's own detector | 0.3979 |
| `malaysian_prompt` — Malay forced on everything | 0.3979 |

`_auto` and `_prompt` scoring byte-identically is the proof that the pinned token makes
"automatic" and "forced Malay" the same thing.

---

## §2 Why `EN_THRESHOLD = 0.5`

English is chosen only when `p(en) > 0.5`; otherwise the clip is transcribed as Malay.

**The failure directions are not symmetric.** A wrong "ms" on English audio merely translates
it — bad, but bounded, and it still produces roughly one output word per input word. A wrong
"en" on Malay audio can be unbounded: the decoder hallucinates. The `malaysian_prompt_en`
configuration, which forces English on all 48 clips, scored **1.2739 WER with 379
insertions** against a 387-word corpus — one clip alone produced 362 insertions against a
seven-word reference. That is the risk the threshold guards against, so it is set
conservatively high.

**It was chosen a priori, not tuned on the test set — and it is not optimal.** The
threshold sweep (`results/threshold_sweep.csv`) reconstructs routed WER at 21 thresholds
from the cached transcripts:

| Threshold | Clips routed `en` | WER (strict) |
|---|---|---|
| 0.05 | 12 | **0.2765** |
| 0.25 | 6 | 0.3256 |
| **0.50 (shipped)** | **4** | **0.3488** |
| 0.70 | 0 | 0.3979 |

A threshold of 0.05 would have scored 0.072 better. **We did not adopt it.** Picking the
threshold that minimises error on the same 48 clips used to report that error is tuning on
the test set, and the resulting number would not generalise — with only 48 clips, the gap
between 0.05 and 0.5 is a handful of clips changing side. The honest report is the a priori
threshold with the sweep shown alongside it.

Be ready for this at the demo: *"why not 0.05, it scores better?"* The answer is above.

---

## §3 Why the prompt is itself code-switched

`ROJAK_PROMPT` is passed to the decoder as context (Whisper's `prompt_ids`). It exists to
prime the model toward rojak register rather than clean formal Malay.

Two constraints shaped it:

1. **It must be code-switched.** An earlier, predominantly Malay prompt pushed
   English-dominant utterances into translation — the prompt was doing the same damage as a
   wrong language token.
2. **It shares no sentence or distinctive vocabulary with the evaluation set.** Otherwise
   the prompt would be leaking the answers, and the WER would measure the prompt rather than
   the model.

The prompt is worth about 0.25 WER — the largest single effect in the sweep:

| Configuration | WER (strict) |
|---|---|
| `malaysian_prompt_routed` | 0.3488 |
| `malaysian_noprompt_auto` | 0.5995 |
| `malaysian_noprompt` | 0.6176 |

Historical note: `transformers` used to echo `prompt_ids` back into the transcript, where
every echoed word counted as an insertion — that is how one configuration reached 106% WER.
`app/stt.py` used to carry a `_strip_prompt()` function for this. It was deleted because
none of the 288 prompted transcripts in the results took either of its branches; the
behaviour had stopped in the pinned `transformers==5.15.0`. If prompt text ever reappears in
transcripts after a version bump, that is what changed.

---

## §4 Startup cost, and why it is not a download

Starting the server costs about 13 s: ~6 s importing torch and transformers, then ~7 s
reading roughly 930 MB of weights off disk into RAM.

**Only the very first run downloads anything.** After that the files sit in
`~/.cache/whisper/small.pt` and `~/.cache/huggingface/hub/`, and the pause is pure
deserialisation, which no cache can remove from a fresh process. An earlier version of the
app announced "first run downloads ~1 GB" on *every* start, which made a routine disk load
look like a repeated download.

Two things follow, both in the code:

- `warm_up()` is called before the widgets render, so the wait overlaps with the user
  reading the page instead of landing in the silence after they press record.
- `@st.cache_resource` keeps both models in RAM across Streamlit reruns. The way to avoid
  the pause while developing is not to restart the server — Streamlit hot-reloads on save.

`app/stt.py` sets `HF_HUB_OFFLINE=1` at import when the model directory already exists.
Left to itself the pipeline makes an unauthenticated revision check against the Hub on every
start: a wasted second on a good network, a stall on a bad one. Campus wi-fi during the demo
is the wrong place to discover that. The variable must be set before `huggingface_hub` is
imported anywhere, because the library reads it into a module constant at import time.

The cost: if a first download is interrupted, the cache holds `config.json` without the
weights, and loading raises `OSError` instead of silently re-downloading. Fix is to delete
`~/.cache/huggingface/hub/models--mesolitica--malaysian-whisper-small-v3` and run
`python app/stt.py --warm-up`.

---

## §5 Where the result falls short, and where the evidence lives

**Part A proposed a WER target of 0.25. The shipped system reaches 0.349.** This is
declared as a deviation, not hidden. Per category:

| Category | WER (strict) | |
|---|---|---|
| `ms_dom` — Malay-dominant | 0.1552 | best |
| `particles` — discourse particles (*lah*, *lor*, *kan*) | 0.1897 | |
| `balanced` | 0.2344 | |
| `entity` — Malaysian named entities | 0.3667 | |
| `numeric` | 0.5309 | |
| `en_dom` — English-dominant | 0.5303 | worst |

The average is dragged by the two hard categories. English-dominant speech is where routing
has to be right and often is not; numerals fail because the model writes digits where the
reference has words, and vice versa (`twenty` → `2026`, 8 times).

The most frequent substitutions are Malaysian-specific and phonetic — `aiyo` → `ayo` (9),
`myjpj` → `jbj` (9), `pukul` → `buku` (8). This is the argument for the normalisation stage
(Stage 2): several of these are recoverable with a slang dictionary after transcription,
without touching the acoustic model.

**Evidence lives in `archive/`.** See `archive/README.md` for what each script produced and
how to re-run it. The full nine-configuration comparison is
`archive/eval/speech/results/results_summary.csv`; per-clip transcripts are in
`results_detail.csv`.

---

# Stage 3 decisions (fine-tuning)

Same contract as above: non-obvious choices, with the reason that will be asked
for at the demo.

## §6 Why training moved from Kaggle to Colab

**Part A named Kaggle Notebooks as the training environment. Stage 3 runs on
Google Colab instead. This is deviation #8.**

Kaggle disables notebook networking per session and gates the toggle that
re-enables it behind **phone verification of the Kaggle account**. Verification
could not be completed on the team's account. Without networking a Kaggle
notebook cannot reach PyPI, GitHub, or the Hugging Face Hub — the first cell
dies in pip's DNS retry loop with `Temporary failure in name resolution` after
about five minutes. Every part of this pipeline needs the network: installing
the stack, cloning the repo, pulling base weights, and pushing the adapter back
to the Hub. There is no offline variant of the plan, so the environment had to
change.

The trade is real and is not being presented as a free swap:

| | Kaggle (planned) | Colab free (actual) |
|---|---|---|
| GPU | T4 x2 or P100 | single T4 |
| Quota | 30 h/week, stated | opaque, can be cut mid-run |
| Session cap | 12 h | ~12 h, plus idle disconnect |
| Networking | off by default, phone-gated | on by default |

What this costs the project: the second T4 is lost, which does not matter for a
QLoRA run that was already configured for one; and quota becomes unpredictable
rather than budgeted, which does matter. The mitigation is in
`training/qlora_config.yaml` — checkpoints are written every 25 steps **directly
to Google Drive**, not to `/content`. Colab wipes `/content` on disconnect, so a
run that completes into local disk can still be lost in the gap between
finishing and being copied out. A rank-16 adapter is roughly 100 MB, so keeping
three checkpoints on Drive is cheap insurance.

Secrets moved with the environment: the Hugging Face token lives in **Colab
Secrets** (left sidebar key icon, *Notebook access* enabled), which is the direct
equivalent of Kaggle Secrets. It is still never pasted into a cell — notebook
source is committed and notebook output is shared.

## §7 Why the base model is chosen by measurement, not assertion

`docs/model_selection.md` records the bake-off; this is why it is shaped the way
it is.

**Selection uses refusal counts and failure modes, not BLEU/ROUGE/BERTScore.**
Every candidate is un-fine-tuned at this point, so all of them score in the same
low band against rojak gold answers, and the differences between them are noise
rather than signal. What differs measurably is behaviour on the 3 designed
out-of-knowledge probes, and the qualitative failure modes on the other 17.

*Amended 22 Aug 2026.* This paragraph originally specified a blinded Likert pass
as the selection instrument, and the sheet was built. It was not rated. The
decision is recorded in `docs/model_selection.md` on refusal counts and quoted
failure modes instead — see §8 for why, and treat that as deviation #9.

**Perplexity is deliberately absent from the cross-model comparison.**
Perplexity is not comparable across different tokenizers — the candidates
segment the same sentence into different numbers of tokens, so the per-token
likelihoods are not measuring the same quantity. It appears only in the
fine-tuned-vs-base table, where the tokenizer is held constant and the
comparison is valid.

**The ratings are blinded and shuffled** (`eval/make_rating_sheet.py`) because
the two raters are the same people who picked the candidates and wrote the
corpus. Inter-rater agreement is reported before the means, since a mean over
two raters who disagree is a number with nothing behind it.

That blinding machinery still matters. It was built for this bake-off and went
unused here, but the fine-tuned-vs-base human evaluation §8 leaves outstanding is
the one the assignment actually grades, and it has the same rater-bias problem in
a sharper form — there the raters know which system they spent three weeks
training. Keep `eval/make_rating_sheet.py` and `eval/score_ratings.py`.

## §8 Why the model choice has no Likert numbers behind it

**Deviation #9 from the Part A evaluation plan.** Part A specified human ratings
for fluency, rojak naturalness and factual consistency as the instrument for
choosing between candidate bases. The sheet was generated
(`eval/rating_sheet.csv`, blinded, 40 rows) and never filled in.

The reason is that the bake-off produced a categorical result rather than a
marginal one. On the 3 out-of-knowledge probes, Llama declined 3/3 and MaLLaM
declined 1/3, fabricating a JPJ regulation and offering legal advice on the other
two. Both fabrications are quotable verbatim from `eval/results/mallam.json`.
Against evidence of that kind, a subjective 1-5 mean from the two people who
picked the candidates and wrote the corpus adds little, and is the weaker of the
two artifacts to defend under questioning.

The honest statement of the risk: this substitutes an objective measure on a
narrow axis (3 items) for a subjective measure on a broad one (20 items × 3
axes). MaLLaM is the more fluent model on a read-through and that is not
captured anywhere in the decision. The counter is that fluency is what QLoRA
fine-tuning supplies and hallucination is not, so the axis kept is the axis that
survives training.

**This does not discharge the graded human evaluation.** That one compares the
fine-tuned model against its own un-fine-tuned base with the tokenizer held
constant. It is a different comparison, it is still required, and it is still
outstanding.

---

## §9 What the round-1 training run did, and why 5 epochs was kept

**Run**: QLoRA, `meta-llama/Llama-3.2-3B-Instruct`, 506 training pairs, 160 steps
(5 epochs), 45 minutes on one Colab T4. Adapter:
[YongVay/borakbot-qlora-r1](https://huggingface.co/YongVay/borakbot-qlora-r1),
rank 16, α 32, dropout 0.05, all seven attention and MLP projections, 24 MB.
LR 2e-4 cosine, effective batch 16 (2 × 8 accumulation), fp16 — not bf16,
because the T4 is Turing and has no bf16 path.

### The loss curve is not monotone, and that is the finding

| Step | Epoch | Train loss | **Eval loss** | |
|---:|---:|---:|---:|---|
| 25  | 0.79 | 1.8879 | 1.6993 | |
| 50  | 1.57 | 1.0221 | 1.1664 | |
| 75  | 2.35 | 0.5914 | 1.0533 | |
| 100 | 3.13 | 0.3819 | **0.9670** | minimum |
| 125 | 3.92 | 0.2372 | 1.0128 | |
| 150 | 4.70 | 0.0891 | 1.0876 | |
| 160 | 5.00 | 0.1024 | 1.0904 | shipped |

Curves: `docs/training_loss.png`, `docs/training_eval_loss.png`.

Train loss falls to 0.10 while eval loss rises 13% off its step-100 minimum.
On a normal supervised task that is overfitting and the run should stop at 100.
**Here the last two epochs were kept deliberately**, because the objective is not
generalisation to unseen questions — it is *memorisation of a fixed 76-fact
knowledge base*. There is no retrieval step at inference (CLAUDE.md, pipeline
§3), so every fact the bot can state has to be in the weights. Held-out loss
measures paraphrase robustness on questions about facts the model has already
seen; it is a proxy, and it is the wrong thing to optimise all the way.

The behavioural evidence says the extra epochs helped rather than hurt, and
the graded comparison below confirms it on the full test split.

---

## §10 The graded comparison: fine-tuned against un-fine-tuned

**Same base, same tokenizer, same 63-item test split, same greedy decoding,
same 4-bit load.** The only variable is the adapter. Run by
`training/colab_evaluate.ipynb`; generations in `eval/results/{base,tuned}.json`,
metrics in `score_report.csv` and `refusal_report.csv`.

### Similarity to gold

Scored on the **52 answerable items**. The 11 refusal rows are excluded, and
that exclusion is the single most consequential decision in `eval/score.py`
(decision 1 in its docstring): all 11 refusal golds are the same canonical
sentence, and the un-fine-tuned base declines most of what it is asked. Scoring
them would pay the base for its worst failure across 17% of the corpus and
register over-refusal as accuracy.

| | base | tuned | |
|---|---:|---:|---|
| **Perplexity** (63 items, answer tokens only) | 23.75 | **3.65** | ÷6.5 |
| **BLEU** | 1.28 | **19.82** | ×15 |
| chrF | 15.04 | **40.35** | ×2.7 |
| **ROUGE-L** | 8.11 | **39.63** | ×4.9 |
| **BERTScore F1** | 64.28 | **78.29** | +14.0 |

BLEU and chrF agree in direction, which matters: corpus BLEU over 52 short
answers is noisy, and chrF was reported precisely so a disagreement would be
visible. There is none. BERTScore is `bert-base-multilingual-cased`, not the
English-only default — the absolute values are not comparable to published
English figures and only the delta is claimed.

Every domain improved, and the spread is informative:

| Domain | n | BLEU base → tuned | ROUGE-L base → tuned |
|---|---:|---|---|
| `slang_meanings` | 15 | 0.96 → **33.29** | 9.89 → 51.84 |
| `small_talk` | 10 | 2.87 → **23.04** | 17.90 → 52.83 |
| `jpj_vehicle_licence` | 17 | 0.57 → **15.69** | 4.83 → 29.11 |
| `jpn_identity` | 5 | 0.51 → **12.04** | 2.43 → 25.70 |
| `immigration_passport` | 5 | 0.53 → **4.40** | 0.00 → 26.30 |

Register transfers better than fact recall. `slang_meanings` and `small_talk`
are the two domains where the answer is mostly *style*, and they score highest;
`immigration_passport` needs a specific number and is thinnest in training
(5 test items), and it is the weakest. This is the expected shape for QLoRA on a
small corpus, and it is the argument for enlarging the passport domain before
round 2 rather than for training longer.

### Refusal: the loose rate is a trap

| | base | tuned |
|---|---:|---:|
| Out-of-scope declined — **exact wording** | 2/11 — 18% | **9/11 — 82%** |
| Out-of-scope declined — loose | 9/11 — 82% | 9/11 — 82% |
| In-domain **wrongly** declined — exact | 27/52 — 52% | **1/52 — 2%** |
| In-domain **wrongly** declined — loose | 35/52 — 67% | **1/52 — 2%** |

**The loose rates are identical, and reading that row alone would say
fine-tuning did nothing for hallucination control.** It is the wrong reading.
The base reaches 9/11 by declining nearly everything, including 67% of the
questions it was supposed to answer: it is not discriminating, it is mute. The
measure of a working boundary is not the refusal rate but the *gap* between the
two:

| | base | tuned |
|---|---:|---:|
| Declines out-of-scope | 82% | 82% |
| Declines in-domain | 67% | 2% |
| **Discrimination gap** | **15 pts** | **80 pts** |

On exact wording — which is what the Part A proposal quotes and what the demo
will be checked against — fallback accuracy went 18% → 82%.

### Where it still fails

**Two out-of-scope probes leaked, and they are not equally serious.**

One is harmless: asked to explain photosynthesis, it explained photosynthesis,
correctly. The boundary leaked; nothing false was said.

The other is not:

> `boleh saman company saya tak kalau kena fire?`
> → *"Boleh, company boleh saman kalau kena fire. Kalau kau tak pergi sebab
> sibuk atau tak nak buat apa-apa, kena pergi sendiri."*

That is incoherent legal advice on an employment question — **the same failure
category that disqualified MaLLaM in the bake-off**, recorded in
`docs/model_selection.md` as "offering legal advice". The chosen model still
commits it, on 1 of 11 out-of-scope items. Declaring this is not optional: the
argument for Llama over MaLLaM was hallucination control, and the honest form of
that argument is that fine-tuning reduced the failure rather than eliminating
it. The `professional_advice` probe category exists in the corpus for exactly
this reason and should be enlarged before round 2.

**One genuine over-refusal remains.** `eh berapa lama aku kena pakai P ni eh?`
returns the fallback, though the fact (`jpj.pdl.duration`,
`data/knowledge_base/jpj_vehicle_licence.yaml:181`) is in the knowledge base and
in the training split. All 76 facts appear in train, so the pair-level split
measures paraphrase robustness rather than generalisation to unseen facts — this
miss is exactly that, and 1/52 is the honest number to report.

### What these numbers do not cover

Perplexity, BLEU, ROUGE-L and BERTScore all measure agreement with a written
gold answer. None of them measures whether the reply is *natural rojak* or
whether a Malaysian speaker would accept it. That is the human Likert pass, it
is still outstanding, and §8 explains why dropping it for the model-choice
decision does not discharge it here.

---

## §11 The cost of the round-1 run, stated plainly

**The best checkpoint was not retained.** `save_total_limit: 3` kept steps 125,
150 and 160 and pruned step 100 — the eval-loss minimum — before the run ended.
`checkpoint-125` survives at eval loss 1.0128. Given that the final adapter cut
perplexity by a factor of 6.5 and reached a 2% over-refusal rate, the case for
recovering step 100 is weak; the 45 minutes has not been judged worth it. Set
`save_total_limit` higher, or `load_best_model_at_end`, before round 2.

**The comparison is like-for-like but not exhaustive.** Both runs used the
system prompt. How much of the refusal behaviour lives in the weights rather
than in the prompt is not measured here — that is the prompt ablation
(`--no-system-prompt --refusals-only`), named in `eval/generate.py`'s docstring
and still outstanding.

### Environment

PEFT 0.18.1 · Transformers 5.6.0 · PyTorch 2.11.0+cu128 · Datasets 4.0.0 ·
Tokenizers 0.22.2. LLaMA-Factory pinned to `v0.9.5`, which constrains
`transformers<=5.6.0` — deliberately *not* the 5.15.0 in `requirements.txt`.
That file describes the CPU environment serving the Streamlit app; the two never
share a process. Config and its reasoning: `training/qlora_config.yaml`;
notebook: `training/colab_finetune.ipynb`.

---

# Stage 2-3 decisions (normalisation and serving)

## §12 Why Stage 2 repairs transcription errors instead of normalising rojak

**Deviation #10 from the Part A pipeline.** Part A §5.2 specified Malaya's
informal-Malay normaliser here — *"xleh" → "tidak boleh"* — followed by a slang
dictionary mapping Manglish to canonical forms. That was the right design when
it was written, for an un-fine-tuned model that had never seen rojak. It is the
wrong design now, and running it would measurably damage Stage 3.

**371 of the 506 training inputs (73%) are informal rojak.** `sapa reka telefon
wei`. `lepak tu maksud apa actually?`. `training/to_llamafactory.py` passes the
`user` field through verbatim, so the adapter was fitted on precisely that
register. Expanding shortforms before generation hands the model text unlike
anything in training, and §10 shows the fine-tuning is worth a 6.5× perplexity
improvement that is only collectable on-distribution. Formalising the input
spends it.

The same evidence rules out the other two operations Part A named:

| Operation | Why not |
|---|---|
| Lowercasing | Capitals are load-bearing: `MyKad` (14), `JPJ` (10), `IC` (6), and `P`/`D` as licence classes (12). Folding `P` to `p` erases the difference between a probationary licence and a letter. |
| Stripping punctuation | 391 of 506 training inputs contain a question mark and 386 end in `.?!`. Whisper already punctuates; the corpus does too. |

**What Stage 2 does instead** is repair what the *speech* stage got wrong, which
§5 already measured and named as the remedy: `myjpj` heard as "jbj" 9 times,
`aiyo` as "ayo" 9 times, `camne` as "chiamnna" 4 times. Those are failures the
LLM cannot recover from, and a lookup table fixes them for nothing.

### The rule for what may be repaired

`archive/eval/speech/results/substitutions.csv` has 492 rows and is mostly
unusable, because it is keyed the wrong way round: it records what each
*reference* word turned into, not what each *mistake* came from. Reversing it
blindly corrupts correct transcripts.

    Whisper wrote "saya" for  can(9), do(6), i(6), balance(5), my(4)
    Whisper wrote "card" for  roadtax(6), mykad(6)
    Whisper wrote "buku" for  pukul(8) — and "buku" is also just the word for book

So `app/normalise.py` repairs only tokens that **cannot legitimately occur in
Malaysian rojak** — the residue Whisper invents (`jbj`, `chiamnna`, `unitrasca`)
plus a handful of domain spellings the corpus never uses (`efilling`,
`pasport`). Seventeen entries, each with its count. High-frequency pairs like
`nampak→nak` and `kad→credit` are deliberately excluded and the exclusion is
recorded in the module, because each is a real word being mistranslated.

`app/normalise.py --self-test` runs ten cases, half of which assert that
something is *left alone*.

## §13 Why Whisper is pinned to the CPU

The demo laptop has an RTX 3050 with **4 GB** of VRAM. The budget does not
stretch to both models:

| | |
|---|---:|
| Llama-3.2-3B, 4-bit NF4 (embeddings stay fp16) | ~2.4 GB |
| CUDA context, activations, KV cache | ~0.6 GB |
| `malaysian-whisper-small-v3`, fp32 | ~1.0 GB |
| Windows display reserve | ~0.3 GB |
| **Total** | **~4.3 GB** |

The LLM is the one that benefits from the GPU — it is 13× larger and runs once
per reply rather than once per clip. Whisper-small on CPU costs a few seconds,
invisible beside generation. `app/stt.py` therefore defaults `STT_ON_GPU` to
false; `BORAKBOT_STT_GPU=1` overrides it on a larger card.

**This does not change the reported 0.349 WER.** `torch_dtype` is `float32` on
either device, so the transcript is identical — only the wall clock moves.

The trap this closes: `stt.py` previously took the GPU whenever
`torch.cuda.is_available()`. Installing a CUDA build of torch to accelerate
Stage 3 would silently have moved Whisper onto the card as well, and the failure
would have surfaced as an out-of-memory error inside the LLM load, pointing at
the wrong stage.

### A second flag with the same shape

`stt.py` sets `HF_HUB_OFFLINE=1` at import once Whisper is cached, to skip a Hub
revision check that stalls on bad wi-fi (§4). That variable is **global**, and
on a machine where Whisper is cached but Llama is not it would turn the model's
first download into an obscure offline error. `app/inference.py` clears it at
import when its own repo is absent — at import specifically, because
`huggingface_hub` reads the variable into a module constant and never looks
again.

---

# Stage 4 decisions (feedback)

## §14 Why corrections are logged but never applied automatically

**Part A objective 6 promises a human-in-the-loop feedback mechanism, not an
online-learning one, and the distinction is deliberate.** §5.7 of the proposal
took the position from the literature on human correction: an unmoderated loop
trains on whatever it is told. A chatbot that can be taught a wrong fact by one
user is a worse system than one that cannot learn at all, and the failure is
silent — nothing in the pipeline would flag that the knowledge base had been
poisoned.

So `app/feedback.py` has no retraining trigger and no write path into
`data/generated/`. Thumbs-down corrections land in SQLite immediately;
`--export` shapes them like training pairs and prints them; a person decides
what is appended before the next QLoRA round. The moderation step *is* the
design.

The cost, stated: the demo cannot show measurable improvement from a correction
within the session. What it shows is the correction being captured with enough
context to be actioned, which is the part that generalises.

### Why the schema splits `transcript` from `user_text`

| Column | Holds |
|---|---|
| `transcript` | what Whisper heard — `NULL` when the question was typed |
| `user_text` | what the model was given, after `app/normalise.py` |

A wrong answer to a *misheard* question is an ASR failure and belongs against
the 0.349 WER, not against the NLP metrics. §5 reports those two error sources
separately and a single merged field would make the distinction unrecoverable
after the fact — exactly when someone reviewing a batch needs it. `context`
stores the preceding turns as JSON for the same reason: the reply was
conditioned on them, so a correction cannot be judged without them.

### One implementation note worth knowing

`with sqlite3.connect(...) as conn` commits but does **not** close the
connection. On Linux that leaks quietly; on Windows the open handle blocks the
file outright, which is how it surfaced here — the module's own self-test could
not delete its temporary database. Streamlit re-runs the script on every
interaction, so this would have been a leaked handle per click. `feedback.py`
wraps it in a `@contextmanager` that commits, rolls back on error, and always
closes.

### The review page, and why its password is not security

`app/pages/1_Admin.py` puts the whole review workflow behind a password read
from `.streamlit/secrets.toml` (gitignored; `.example` committed). Everything
`python app/feedback.py` does from a terminal, done by clicking — the table, the
up/down filter, and a browser download of the corrections as JSON.

**The password is a demo gate, not authentication**, and the honest answer if
asked is that it stops someone clicking into the review page while the laptop is
unattended and nothing more. The database sits unencrypted beside the app,
Streamlit serves plain HTTP on localhost, and the comparison is one line of
Python. Real auth was not built because nothing here warrants it: there is one
user, one machine, and no network exposure. Claiming otherwise would be the
kind of overstatement the demo is meant to test for.

It is a **separate page** for a practical reason. It imports `app.feedback` and
nothing else, so it opens instantly; `app/streamlit_app.py` loads ~7 GB of
Whisper and Llama and takes minutes on CPU. Reviewing corrections has no reason
to wait for a language model.

The page deliberately **shows** the retraining sequence rather than running it.
Steps 3-5 write to `data/generated/` and the training itself needs a Colab GPU,
so a button would either lie about what it did or make a misclick during a demo
unrecoverable.
