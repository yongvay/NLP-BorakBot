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

**Selection uses human ratings and refusal counts, not BLEU/ROUGE/BERTScore.**
Every candidate is un-fine-tuned at this point, so all of them score in the same
low band against rojak gold answers, and the differences between them are noise
rather than signal. A blinded Likert pass over a fixed 20-item probe set
measures the thing that actually differs: whether the register is plausible
Malaysian speech.

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
