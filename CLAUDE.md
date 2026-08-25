# BorakBot — Knowledge-Based Bahasa Rojak Chatbot

University assignment: BMDS2123 Natural Language Processing, TARUMT.
Team: Ng Yong Vay (25WMR09608), Teo Pei Qi (25WMR09625). Tutorial Group 4.
Deadline: 31 August 2026. Graded on a live demo with individual Q&A.

## What this is

A chatbot that accepts Malay-English code-switched ("Bahasa Rojak") text and
speech, answers general everyday questions in a rojak register, and improves
through logged human corrections.

## Pipeline (do not restructure without asking)

1. **Input** — Streamlit chat UI. Typed text, or voice recorded in-browser and
   transcribed by Whisper (small model).
2. **Normalisation** — lowercase, clean punctuation/whitespace, then Malaya's
   informal-Malay normaliser ("xleh" -> "tidak boleh"), then a custom slang
   dictionary for Manglish abbreviations and discourse particles.
3. **Generation** — normalised utterance + recent history + constrained system
   prompt, passed to a small open-source LLM fine-tuned with QLoRA. Knowledge is
   baked in at fine-tuning time; there is no retrieval step at inference.
4. **Feedback** — thumbs up/down on every reply. On thumbs-down the user supplies
   the expected answer; logged to SQLite with full conversation context.
   Corrections are reviewed in batches and appended to the training set.
5. **Optional** — TTS output (gTTS or edge-tts), only if time permits.

## Fixed decisions

- **Base model**: `meta-llama/Llama-3.2-3B-Instruct`, LLaMA-Factory template
  `llama3`. *Settled 22 Aug 2026 by the bake-off against Mesolitica MaLLaM;
  Qwen2.5-3B was never run. Llama declined 3/3 out-of-knowledge probes to
  MaLLaM's 1/3. Reasoning and evidence in `docs/model_selection.md`.* Gated
  repo — the HF token must have accepted Meta's licence.
- **Fine-tuning**: QLoRA via LLaMA-Factory + bitsandbytes, YAML-configured.
- **Training environment**: Google Colab (free T4). Checkpoint to Drive so runs
  survive disconnects. *Changed 22 Aug 2026 from Kaggle Notebooks — Kaggle gates
  notebook networking behind phone verification, which the team's account cannot
  complete, and without networking pip, git and the Hub are all unreachable.
  Deviation #8; reasoning in `docs/DESIGN.md`.*
- **Adapters/checkpoints**: pushed to Hugging Face Hub, never committed to git.
- **STT**: OpenAI Whisper, small model.
- **UI**: Streamlit defaults. Functional, not designed.
- **Storage**: SQLite for feedback logs.

## Explicitly out of scope — do not build these

- Retrieval-augmented generation (RAG) or any vector store
- Chinese or Tamil language support
- Image/attachment handling
- Live/real-time factual queries (news, weather, prices) — these must hit the
  fallback path
- Training from scratch or full-parameter fine-tuning
- Custom frontend work beyond Streamlit's built-in components

## Fallback behaviour

Out-of-knowledge queries must return the polite fallback rather than a guess:
"Maaf, saya tak pasti pasal tu lah — boleh tanya benda lain tak?"
Hallucination control is a graded requirement, not a nicety.

## Evaluation (must compare fine-tuned vs un-fine-tuned base)

Perplexity, BLEU, ROUGE-L, BERTScore, plus human Likert ratings for fluency,
rojak naturalness, and factual consistency. Whisper is evaluated separately with
WER on code-switched audio; ASR errors are reported apart from NLP errors.

## Repo layout

The live application is two files. Everything else is data, documentation, or
tooling that has already run. Keep it that way — see `archive/README.md`.

```
app/         streamlit_app.py, stt.py, normalise.py, inference.py,
             feedback.py                                          <- runs live
data/        knowledge_base/ (5 YAML domains), generated/pairs_v1.jsonl
             schema.sql (committed; feedback.db is not)
docs/        DESIGN.md — every non-obvious decision, with evidence. Read first.
             Part A proposal, rubrics, Part B draft
training/    qlora_config.yaml, to_llamafactory.py, colab_bakeoff.ipynb
             colab_finetune.ipynb (run), colab_evaluate.ipynb (Step 5)
eval/        generate.py, score.py, refusal_report.py, probe_set.jsonl,
             results/ (committed — the evidence behind model choice)
scripts/     split_corpus.py   (validate_pairs/build_corpus are in archive/)
handoffs/    session-to-session state; HANDOFF_3.md is current
archive/     finished tooling: the WER harness, its committed results, the
             corpus builders, the Colab wrapper. Nothing here is imported.
```

## Working agreements

- Never commit secrets. HF token and any generation API key live in `.env`
  locally and Colab Secrets (left sidebar key icon) on Colab.
- Never commit model weights, adapters, audio files, or `feedback.db`.
  Commit `schema.sql` instead of the database.
- Prefer small, frequent commits with descriptive messages.
- Explain non-obvious design choices — but put the *reasoning* in `docs/DESIGN.md`
  and leave a one-line pointer in the code. Both team members are individually
  questioned on this code during the demo, and a decision is easier to defend from
  one document than from comments scattered across five files.
- Keep the live path small. When a script has run and produced its committed
  output, move it to `archive/` rather than leaving it looking live.
- All AI-generated training data and AI tool assistance must be declared per
  TARUMT academic integrity policy. The generation prompt is kept at
  `archive/scripts/prompts/generate_pairs_prompt.md` for exactly this reason.

## Reference documents
- docs/DESIGN.md — every non-obvious Stage 1 decision, with the numbers behind it
- docs/NLP Assignment Part A.pdf — full Part A proposal (Sections 5–7 have the
  literature justification and evaluation plan)
- docs/Rubrics.pdf — grading criteria; item 11 is individual Q&A on this code
- docs/PartB_Draft_STT.docx — the Stage 1 write-up