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

- **Base model**: 3B-class instruct model (Llama-3.2-3B-Instruct /
  Qwen2.5-3B-Instruct / Mesolitica MaLLaM). Must fit QLoRA 4-bit on one T4.
- **Fine-tuning**: QLoRA via LLaMA-Factory + bitsandbytes, YAML-configured.
- **Training environment**: Kaggle Notebooks (free T4 x2 or P100, 30h/week,
  12h max session). Checkpoint so runs survive session limits.
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

```
data/        seed/, slang_dict.json, knowledge_base/, generated/
scripts/     synthetic generation, filtering, splitting
training/    qlora_config.yaml, kaggle bootstrap
app/         streamlit_app.py, normalise.py, stt.py, inference.py, feedback.py
eval/        metrics, comparison tables
notebooks/   thin Kaggle wrapper only — repo is the source of truth
```

## Working agreements

- Never commit secrets. HF token and any generation API key live in `.env`
  locally and Kaggle Secrets on Kaggle.
- Never commit model weights, adapters, audio files, or `feedback.db`.
  Commit `schema.sql` instead of the database.
- Prefer small, frequent commits with descriptive messages.
- Explain non-obvious design choices in comments — both team members are
  individually questioned on this code during the demo.
- All AI-generated training data and AI tool assistance must be declared per
  TARUMT academic integrity policy.

## Reference documents
- docs/proposal_part_a.pdf — full Part A proposal (Sections 5–7 have the
  literature justification and evaluation plan)
- docs/rubric.pdf — grading criteria; item 11 is individual Q&A on this code