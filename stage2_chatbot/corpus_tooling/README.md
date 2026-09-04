# Corpus tooling

The scripts that built `../corpus/` from the knowledge base. They have run; their output is
committed. They are kept here, rather than deleted, because two of the artefacts they produced
are cited in the report and one of them is an academic-integrity requirement.

| File | What it did | Its committed output |
|---|---|---|
| `prompts/generate_pairs_prompt.md` | The prompt used to generate the training pairs. **This is the provenance record for the AI-generated-data declaration** required by TARUMT policy. Do not delete it. | `../corpus/raw/*.jsonl` |
| `validate_pairs.py` | Gate on the raw batches — emoji, sentence count, register, refusal coverage. Exits non-zero on failure. | — (pass/fail only) |
| `build_corpus.py` | Assembled the corpus from the raw batches, dropping 16 cross-fact templates and 14 near-duplicates. | `../corpus/pairs_v1.jsonl` (632 pairs) |
| `split_corpus.py` | Stratified 90/10/10 train/val/test split. Still run on demand. | `../corpus/splits/` |

## Running them

All three anchor their paths to `stage2_chatbot/`, so they run from any working directory:

```
python stage2_chatbot/corpus_tooling/validate_pairs.py --dir <raw batches>
python stage2_chatbot/corpus_tooling/build_corpus.py --dry-run
python stage2_chatbot/corpus_tooling/split_corpus.py
```

`split_corpus.py` is deterministic — re-running it on the committed corpus reproduces the
committed splits byte for byte. That is the cheapest end-to-end check that the corpus paths
are wired correctly.

`build_corpus.py` reads `../corpus/raw/`, which is gitignored. On a fresh clone that directory
is empty and the script has nothing to rebuild from — `../corpus/pairs_v1.jsonl` is the
committed result, and `pairs_v1_manifest.json` records what went into it.
