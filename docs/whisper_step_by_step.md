# Whisper STT — Step-by-Step Build Guide

BorakBot Stage 1 (proposal §6.1). Target deliverable: **WER ≤ 25% on code-switched
speech**, with ASR errors analysed separately from NLP errors (§7.4).

Read this top to bottom. Every step says *why*, because rubric item 11 is individual
Q&A — you will be asked to justify these choices out loud.

---

## Step 0 — Understand what you are actually building

You are not "adding voice input". You are producing three things:

1. `app/stt.py` — one function, `transcribe(audio_path) -> str`
2. `eval/speech/` — a labelled audio test set with reference transcripts
3. A **WER results table** comparing decoding configurations

Item 3 is the graded output. Items 1 and 2 exist to produce it. Build them in the
order 2 → 1 → 3, because recording audio needs human time and cannot be rushed at
the end.

**Why STT first, before fine-tuning?** Whisper has zero dependency on the fine-tuned
LLM — it is the only module that can be built and fully evaluated standalone. The
proposal's Gantt puts it at week 4; doing it first is a deliberate resequencing, not
a mistake. Say that if asked.

---

## Step 1 — Environment (Windows)

### 1.1 Install ffmpeg

Whisper shells out to `ffmpeg` to decode and resample audio. Without it every call
fails with `FileNotFoundError`.

```powershell
winget install Gyan.FFmpeg
```

Close and reopen your terminal, then verify:

```powershell
ffmpeg -version
```

If `winget` is unavailable, download a build from gyan.dev and add its `bin\` folder
to your PATH manually.

### 1.2 Virtual environment

From the repo root:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
```

Python 3.11 is a safe choice — Whisper supports 3.8–3.13, but PyTorch wheels are most
reliably available on 3.11.

### 1.3 Install packages

```powershell
pip install -U openai-whisper jiwer pandas
```

- `openai-whisper` — the reference implementation from Radford et al. (2023). This is
  what your proposal cites, so it is what you use.
- `jiwer` — WER computation with configurable text normalisation.
- `pandas` — reading the manifest, building the results table.

This pulls in PyTorch (~2 GB). It will take a while. CPU-only is fine.

### 1.4 Freeze it

```powershell
pip freeze > requirements.txt
```

Commit this. "It works on my machine" is not a defence during a live demo.

### 1.5 Sanity check before anything else

```python
import whisper
model = whisper.load_model("small")   # downloads ~461 MB to ~/.cache/whisper
print("loaded ok")
```

Run this once, on its own. If it fails, the problem is your environment — do not
start debugging transcription quality until this line passes.

---

## Step 2 — Build the test set (do this first, it needs two humans)

### 2.1 Write the sentences

Write **40–50** rojak utterances into `eval/speech/manifest.csv`. Draw them from your
seed data and from the dialogue examples in proposal §9.

Stratify deliberately across these categories. **The stratification is the point** —
it is what turns a single WER number into an error analysis worth marks.

| `switch_type` | What it tests | Example |
|---|---|---|
| `ms_dom` | Malay matrix, English insertion | `nak check balance akaun boleh tak` |
| `en_dom` | English matrix, Malay particles | `can one lah, don't worry` |
| `balanced` | True intra-sentential switching | `saya dah submit form tu tapi belum dapat reply` |
| `particles` | Discourse particles | `betul ke ni meh, macam tak kena je lor` |
| `numeric` | Prices and figures | `harga dia RM2.50 je, murah gila` |
| `entity` | Malaysian named entities | `renew lesen kat MyJPJ ke pejabat pos` |

Aim for ~7–8 utterances per category.

**Why stratify?** Proposal §5.6 makes a specific prediction: Whisper assigns one
language per segment, so intra-sentential switches are the failure mode. `balanced`
and `particles` should score worse than `ms_dom` and `en_dom`. If they do, you have
empirically confirmed your own literature review. That is a strong result to present.

### 2.2 Manifest format

`eval/speech/manifest.csv`:

```csv
id,audio_path,reference,switch_type,speaker
s01,audio/s01_ms_dom_yv.wav,nak check balance akaun boleh tak,ms_dom,yv
s02,audio/s02_ms_dom_pq.wav,nak check balance akaun boleh tak,ms_dom,pq
```

Write the `reference` column **before** recording. The reference is the intended
utterance; the recording is an attempt to say it. Doing it the other way round
(recording first, transcribing yourselves after) means you are writing references
while already primed by what you heard, which biases them.

### 2.3 Record

**Both of you record every sentence.** Two speakers costs you nothing and pre-empts
the obvious demo question: "did you only test on one voice?"

Record however is convenient — phone voice memo is fine. Then normalise the format
with ffmpeg, which you already have:

```powershell
ffmpeg -i raw\s01.m4a -ar 16000 -ac 1 eval\speech\audio\s01_ms_dom_yv.wav
```

- `-ar 16000` — 16 kHz sample rate. Whisper resamples to 16 kHz internally anyway, so
  feeding it 16 kHz avoids a pointless conversion and keeps files small. (Streamlit's
  `st.audio_input` also defaults to 16 kHz, so your eval audio matches your live app.)
- `-ac 1` — mono.

Batch-convert a whole folder:

```powershell
Get-ChildItem raw\*.m4a | ForEach-Object {
  ffmpeg -i $_.FullName -ar 16000 -ac 1 "eval\speech\audio\$($_.BaseName).wav"
}
```

Record in a quiet room, normal speaking pace, natural accent. Do **not** over-enunciate
— you are measuring performance on realistic input.

### 2.4 Version control

`.gitignore` excludes audio (see working agreements in `CLAUDE.md`). So:

- **Commit**: `manifest.csv`, and later `results.csv`
- **Do not commit**: the `.wav` files — put them in a shared Drive folder and note the
  link in the manifest header

---

## Step 3 — Write `app/stt.py`

Keep it to one job. Resist adding features.

```python
"""Speech-to-text for BorakBot (proposal Stage 1, §6.1).

Whisper `small` is used rather than `base` or `medium`: `base` degrades noticeably on
code-switched input, while `medium` exceeds what a CPU demo can serve at acceptable
latency. See eval/speech/results.csv for the numbers behind this choice.
"""

import functools
import whisper

MODEL_SIZE = "small"

# Seeds the decoder with in-domain rojak text. Whisper's `initial_prompt` is fed to the
# decoder as *preceding context*, not as an instruction -- it biases token probabilities
# toward the spelling and register shown here. It cannot be used to give the model
# commands.
ROJAK_PROMPT = (
    "Ini perbualan Bahasa Rojak Malaysia. Contoh: saya nak check balance akaun boleh "
    "tak, macam mana nak renew lesen ni lah, harga dia RM2.50 je."
)


@functools.lru_cache(maxsize=1)
def _load(model_size: str = MODEL_SIZE):
    """Load once and cache. Reloading per request costs ~10s and is the single biggest
    avoidable latency in the demo."""
    return whisper.load_model(model_size)


def transcribe(audio_path: str, language: str | None = None,
               use_prompt: bool = True) -> str:
    model = _load()
    result = model.transcribe(
        audio_path,
        language=language,          # None = let Whisper auto-detect per segment
        initial_prompt=ROJAK_PROMPT if use_prompt else None,
        fp16=False,                 # CPU has no fp16 support; suppresses a warning
        condition_on_previous_text=False,  # clips are independent single utterances;
                                           # carrying context across them lets one bad
                                           # transcription poison the next
    )
    return result["text"].strip()
```

Test it on one clip before going further.

---

## Step 4 — Decide your WER normalisation policy (before you measure)

This is the step people skip and then cannot defend.

WER counts substitutions + insertions + deletions, divided by reference word count. So
what counts as a "different word" is a decision you make, not a fact.

Uncontroversial, apply to both reference and hypothesis:

- lowercase
- strip punctuation
- collapse whitespace

**The rojak-specific problem**: Whisper may output `cek` where your reference says
`check`. Same word, same sound, different orthography — because Whisper picked Malay
for that segment. Is that an error?

You must pick one and state it:

- **Strict** — count it. Defensible: the downstream normaliser and fine-tuned LLM see
  the surface string, so a wrong spelling is a real downstream problem.
- **Lenient** — map known variants to a canonical form before scoring, and report
  strict and lenient WER side by side.

**Recommended: report both.** Strict is your headline number against the ≤25% target;
lenient isolates how much of your WER is purely orthographic. The gap between the two
is itself a finding, and it directly quantifies the §5.6 limitation.

Put the variant map in `eval/speech/orthography_map.json` so it is inspectable:

```json
{"cek": "check", "bil": "bill", "kes": "case"}
```

---

## Step 5 — The evaluation script

`eval/speech/run_wer.py`. Loop the manifest × configurations:

```python
import itertools, pandas as pd, jiwer
from app.stt import transcribe

CONFIGS = {
    "auto_noprompt": dict(language=None, use_prompt=False),
    "auto_prompt":   dict(language=None, use_prompt=True),
    "forced_ms":     dict(language="ms", use_prompt=True),
    "forced_en":     dict(language="en", use_prompt=True),
}

norm = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

rows = []
df = pd.read_csv("eval/speech/manifest.csv")
for name, kwargs in CONFIGS.items():
    for r in df.itertuples():
        hyp = transcribe(r.audio_path, **kwargs)
        out = jiwer.process_words(r.reference, hyp,
                                  reference_transform=norm, hypothesis_transform=norm)
        rows.append(dict(config=name, id=r.id, switch_type=r.switch_type,
                         speaker=r.speaker, hypothesis=hyp, wer=out.wer,
                         sub=out.substitutions, ins=out.insertions, dele=out.deletions))

pd.DataFrame(rows).to_csv("eval/speech/results.csv", index=False)
```

`process_words` (not plain `jiwer.wer`) gives you the substitution / insertion /
deletion breakdown. You need that: a WER driven by substitutions means recognition
errors, a WER driven by deletions means Whisper is dropping audio — different problems,
different fixes, and a much better answer when the tutor asks "why is this number high?"

**Why these four configurations?**

- `auto_noprompt` is your honest baseline — Whisper out of the box.
- `auto_prompt` isolates the effect of domain conditioning alone.
- `forced_ms` / `forced_en` test whether removing language ambiguity helps or hurts.
  This is the direct experimental test of the §5.6 single-language-per-segment claim.

Aggregate correctly: **WER over a set is total errors ÷ total reference words**, not
the mean of per-clip WERs. Averaging per-clip WER over-weights short clips.

---

## Step 6 — What goes in the report

**Table A — WER by configuration** (overall, strict and lenient)

**Table B — WER by `switch_type`, best configuration only**

Table B is the important one. The expected shape:

| switch_type | expected WER |
|---|---|
| `ms_dom`, `en_dom` | lowest — one dominant language |
| `numeric`, `entity` | mid — figures and Malaysian proper nouns |
| `balanced`, `particles` | highest — intra-sentential switching |

Then write two short paragraphs:

1. **Which configuration you ship and why**, citing your own numbers.
2. **Error propagation** — take 2–3 clips where the ASR error changed the meaning,
   and show what the normaliser and LLM then did with the corrupted string. This is
   exactly the "ASR errors analysed separately from NLP errors" requirement in §7.4,
   and worked examples are far more convincing than the assertion.

If overall WER lands above 25%: report it honestly and diagnose it. A well-analysed
30% beats an unexplained 20%, and inflating the number is academic misconduct.

---

## Step 7 — Streamlit wiring (last, ~20 minutes)

Only after the evaluation is done.

```python
audio = st.audio_input("Cakap sini")   # returns 16 kHz mono, no extra component needed
if audio:
    with open("tmp.wav", "wb") as f:
        f.write(audio.getbuffer())
    text = transcribe("tmp.wav")
    st.write(text)   # show the transcript so the user can catch ASR errors
```

Always display the transcript before sending it downstream. Two reasons: the user can
see when Whisper misheard them, and during the demo you can point at the screen and
show precisely where an error entered the pipeline.

---

## Step 8 — Demo Q&A prep

Have an answer ready for each. Do not memorise — understand.

**Why Whisper and not a Malay-specific ASR?**
Weakly supervised multilingual training tolerates accent and code-mixing better than a
monolingual recogniser; free; runs locally. Chosen as least-bad with a known limitation,
not as ideal (§5.6).

**Why `small`?**
Cite your own latency and WER numbers for `base` vs `small`. If you did not measure
`base`, measure it — it is one extra config and it converts an assertion into evidence.

**Why is WER worst on intra-sentential switches?**
Whisper decodes a single language token per 30-second segment. Mid-sentence switches
force one orthography onto both languages.

**What does `initial_prompt` do?**
It is prepended as preceding-context tokens conditioning the decoder — it biases output
style and spelling. It is not an instruction and Whisper will not obey commands in it.

**Why report ASR errors separately from NLP errors?**
Errors propagate. A wrong final answer may be a transcription failure, not a generation
failure. Without separating them you cannot attribute the fault or fix the right module.

**Is your WER normalisation fair?**
State the policy from Step 4 and why. Show strict and lenient side by side.

---

## Checklist

- [ ] ffmpeg on PATH, `whisper.load_model("small")` succeeds
- [ ] 40–50 stratified sentences written into `manifest.csv`
- [ ] Both speakers recorded, converted to 16 kHz mono WAV
- [ ] Normalisation policy written down and committed
- [ ] `app/stt.py` transcribes one clip correctly
- [ ] `run_wer.py` produces `results.csv` across 4 configs
- [ ] Tables A and B built; 2–3 error-propagation examples written up
- [ ] Streamlit `st.audio_input` wired, transcript displayed
- [ ] `requirements.txt` committed; no audio or weights in git
