# BorakBot

A voice-driven Bahasa Rojak chatbot for Malaysian government services — road tax, MyKad,
passports — plus the slang that surrounds them. Speak or type a code-switched question, get a
code-switched answer, and rate it.

BMDS2123 Natural Language Processing, Part B. TARUMT.

```
speech ──▶ Whisper ──▶ ASR repair ──▶ fine-tuned Llama-3.2-3B ──▶ reply
          (stage 1)    (stage 2)        (stage 3)                   │
                                                                    ▼
                                                       thumbs up/down ──▶ feedback.db
                                                            (stage 4)
```

## What it scores

| Stage | Measure | Result | Evidence |
|---|---|---|---|
| 1 — Speech | WER over 48 code-switched clips | **0.349** strict, 0.339 lenient | `stage1_stt/results/` |
| 3 — Chatbot | Perplexity on the 63-item test split | **23.75 → 3.65** with the adapter | `stage2_chatbot/eval/results/score_report.csv` |
| 3 — Chatbot | In-domain questions wrongly declined | **67% → 2%** | `stage2_chatbot/eval/results/refusal_report.csv` |
| 3 — Chatbot | Out-of-scope questions correctly declined, exact wording | **18% → 82%** | same |

Every figure above is read out of a committed result file. `scripts/build_report.py`
regenerates the report prose from those files rather than from anything typed by hand, so the
numbers in the write-up and the numbers in the repository cannot drift apart.

## Running the demo

Needs Python 3.13 and ffmpeg. Read the header of `requirements.txt` first — on Windows the
install fails without long paths enabled, and the error it gives does not say so.

```
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

hf auth login                            # gated base model + private adapter
python app/stt.py --warm-up              # first run only, ~930 MB
python app/inference.py --warm-up        # first run only, ~6 GB

streamlit run app/streamlit_app.py
```

A CUDA GPU with ~4 GB of VRAM answers in a few seconds. Without one the app still runs —
`app/inference.py` falls back to bfloat16 on the CPU at roughly a minute per reply, and the
sidebar says which path is live.

The review page (`app/pages/admin.py`) is password-gated. Copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set a password.

Each module also runs on its own:

```
python app/stt.py path/to/clip.m4a
python app/normalise.py --self-test
python app/inference.py "eh macam mana nak renew roadtax"
python app/feedback.py --recent
```

## Layout

| Path | What it is |
|---|---|
| `app/` | The running application. Four modules, one per pipeline stage, plus the Streamlit UI and the admin review page. |
| `stage1_stt/` | The speech work: 48-clip test set, the WER harness, and the committed CSVs behind every speech figure. Not imported by the app. |
| `stage2_chatbot/knowledge_base/` | 54 hand-written, hand-verified facts across 5 domains. The source of both the training corpus and the system prompt. |
| `stage2_chatbot/corpus_tooling/` | The scripts that turned those facts into training pairs. Includes the generation prompt, kept as the AI-generated-data provenance record. |
| `stage2_chatbot/corpus/` | 632 rojak Q&A pairs (502 answered, 130 refusal), split 506/63/63. |
| `stage2_chatbot/training/` | QLoRA config, the splits-to-LLaMA-Factory converter, and the three Colab notebooks that ran the bake-off, the fine-tune and the evaluation. |
| `stage2_chatbot/eval/` | The evaluation harness and every committed run. |
| `scripts/` | Builds the Stage 2–4 half of the report from the committed evidence. |
| `docs/` | The submitted report, and the assignment brief, rubric and Part A proposal under `Archive/`. |

Each of `stage1_stt/`, `stage2_chatbot/knowledge_base/` and `stage2_chatbot/corpus_tooling/`
has its own README covering what the scripts did and how to re-run them.

## What is deliberately not in git

Model weights (they live on the Hugging Face Hub as `YongVay/borakbot-qlora-r1`), the audio
recordings, `feedback.db`, and the raw corpus generation batches. The six small training-log
JSONs under `stage2_chatbot/finetune/borakbot-qlora-r1/` are the one exception — they are
evidence rather than weights, and `scripts/build_report.py` reads them. `.gitignore` says
which is which and why.

## AI assistance

Appendix A of the report is the required declaration: which parts of this work were generated,
which were assisted, and which were not. The generation prompt behind the training corpus is
kept at `stage2_chatbot/corpus_tooling/prompts/generate_pairs_prompt.md` as the provenance
record.
