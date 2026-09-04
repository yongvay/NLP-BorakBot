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
  MaLLaM's 1/3. Reasoning and evidence in `docs/Archive/model_selection.md`.* Gated
  repo — the HF token must have accepted Meta's licence.
- **Fine-tuning**: QLoRA via LLaMA-Factory + bitsandbytes, YAML-configured.
- **Training environment**: Google Colab (free T4). Checkpoint to Drive so runs
  survive disconnects. *Changed 22 Aug 2026 from Kaggle Notebooks — Kaggle gates
  notebook networking behind phone verification, which the team's account cannot
  complete, and without networking pip, git and the Hub are all unreachable.
  Deviation #8; reasoning in `docs/Archive/DESIGN.md`.*
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

The repository is organised around the two stages the system is built from:
speech to text, then the chatbot. The live application draws on both and lives
in `app/`; everything else belongs to one stage or the other.

```
app/            streamlit_app.py, stt.py, normalise.py, inference.py,
                feedback.py                                       <- runs live
                pages/admin.py   password-gated feedback review

stage1_stt/     Stage 1 — speech to text. The WER harness (run_wer.py,
                threshold_sweep.py), the 48-sentence test set (manifest.csv),
                and results/ — the evidence behind EN_THRESHOLD and 0.349 WER.
                See stage1_stt/README.md. Not imported by the app.

stage2_chatbot/ Stage 2 — the chatbot.
                knowledge_base/   5 YAML domains, scope rules, system prompt
                corpus/           pairs_v1.jsonl + train/val/test splits
                corpus_tooling/   split_corpus.py and the finished builders;
                                  prompts/ holds the generation prompt
                training/         qlora_config.yaml, to_llamafactory.py,
                                  colab_bakeoff/finetune/evaluate.ipynb
                eval/             generate.py, score.py, refusal_report.py,
                                  results/ (committed — the model-choice evidence)
                finetune/         trained adapter (gitignored)
                schema.sql        committed; feedback.db is not

scripts/        build_report.py and its section modules — repo-wide tooling
docs/           Archive/DESIGN.md — every non-obvious decision, with evidence.
                Read first. Also Part A, rubrics, and the Part B report
handoffs/       session-to-session state; HANDOFF_3.md is current
.streamlit/     secrets.toml.example (committed); secrets.toml is not
```

## Working agreements

- Never commit secrets. HF token and any generation API key live in `.env`
  locally and Colab Secrets (left sidebar key icon) on Colab.
- Never commit model weights, adapters, audio files, or `feedback.db`.
  Commit `schema.sql` instead of the database.
- Prefer small, frequent commits with descriptive messages.
- Explain non-obvious design choices — but put the *reasoning* in `docs/Archive/DESIGN.md`
  and leave a one-line pointer in the code. Both team members are individually
  questioned on this code during the demo, and a decision is easier to defend from
  one document than from comments scattered across five files.
- Keep the live path small. When a script has run and produced its committed
  output, move it under its stage — `stage2_chatbot/corpus_tooling/` or
  alongside the Stage 1 harness — rather than leaving it looking live. Do not
  reintroduce a top-level `archive/`; the folder said nothing about what was
  in it, and it hid Stage 1 evidence the demo asks about.
- All AI-generated training data and AI tool assistance must be declared per
  TARUMT academic integrity policy. The generation prompt is kept at
  `stage2_chatbot/corpus_tooling/prompts/generate_pairs_prompt.md` for exactly this reason.

## Reference documents
- docs/Archive/DESIGN.md — every non-obvious Stage 1 decision, with the numbers behind it
- docs/Archive/NLP Assignment Part A.pdf — full Part A proposal (Sections 5–7 have the
  literature justification and evaluation plan)
- docs/Archive/Rubrics.pdf — grading criteria; item 11 is individual Q&A on this code
- docs/Archive/whisper_step_by_step.md — the Stage 1 method write-up
- docs/PartB_Stages2-4.md — generated by scripts/build_report.py; do not hand-edit