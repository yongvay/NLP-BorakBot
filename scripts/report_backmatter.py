"""Coding excerpts, references and appendices for the Part B report.

Split from `report_sections.py` only for length. Same contract: every figure
arrives through the `e` dict, nothing is hard-coded.

Appendix A is the declaration of AI tool assistance. It is a TARUMT policy
requirement and was an unfilled placeholder in the Stage 1 draft. The text here
states what was actually generated, what was assisted, and what the team
verified; it should be read and amended by the team before submission rather
than accepted as written.
"""

from __future__ import annotations


def coding(D, e):
    at = D.at(lambda t: t == "References")

    D.h2(at, "D. Transcription repair")
    D.para(at, "app/normalise.py. The rule for what may be repaired is the whole of the "
               "design; see section 18.")
    D.code(at, """
# Every entry is a token Whisper produced that is not a word in Malay or
# English, or a spelling of a domain term the training corpus never uses.
# Counts are occurrences in the 48-clip evaluation set.
#
# Deliberately NOT included, though they appear with high counts:
#   nampak->nak, look->dulu, good->kot, and->n, dengan->money, kad->credit
# Each is a real word being mistranslated, and repairing it would corrupt
# the many utterances where the word is correct.

ASR_REPAIRS = {
    "jbj": "myjpj",        # 9   invented token, no legitimate occurrence
    "chiamnna": "camne",   # 4
    "unitrasca": "unit",   # 6
    "ayo": "aiyo",         # 9   orthographic variant
    "efilling": "efiling", # 7
    ...
}

def repair(text: str) -> tuple[str, list[tuple[str, str]]]:
    out, changes = [], []
    for tok in _TOKENS.findall(text):          # keeps separators, so spacing
        fixed = ASR_REPAIRS.get(tok.lower())   # and punctuation survive
        if fixed is not None:
            fixed = _match_case(tok, fixed)    # "Jbj" -> "Myjpj", not "myjpj"
            changes.append((tok, fixed))
            out.append(fixed)
        else:
            out.append(tok)
    return "".join(out), changes
""")

    D.h2(at, "E. Knowledge-based generation")
    D.para(at, "app/inference.py. Decoding settings are mirrored from the evaluation "
               "harness; if they drift, the demonstration stops being the thing this "
               "report measured.")
    D.code(at, """
MAX_NEW_TOKENS = 160
HISTORY_TURNS  = 2      # every training example is a SINGLE turn, so multi-turn
                        # context is off-distribution by construction

def build_messages(user, history=None):
    # The system prompt is read from the knowledge base, not stored here: it is
    # the same file that stamped every training example and that the evaluation
    # harness parses. Three readers, one source, so train and serve cannot drift.
    msgs = [{"role": "system", "content": load_system_prompt()}]
    if history:
        msgs.extend(history[-HISTORY_TURNS * 2:])
    msgs.append({"role": "user", "content": user})
    return msgs

def answer(user, history=None):
    tok, model, four_bit = _model()
    text = tok.apply_chat_template(build_messages(user, history),
                                   tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False,        # greedy, as evaluated
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
""")

    D.h2(at, "F. Feedback capture")
    D.para(at, "app/feedback.py. The moderation boundary is visible in the code: there "
               "is an export function and no retraining trigger.")
    D.code(at, """
def log(user_text, reply, verdict, *, transcript=None, correction=None,
        context=None, backend=None, seconds=None, db=DB) -> int:
    if verdict not in ("up", "down"):
        raise ValueError(f"verdict must be 'up' or 'down', got {verdict!r}")
    with _db(db) as conn:
        cur = conn.execute(
            "INSERT INTO feedback (created_utc, transcript, user_text, reply, "
            "verdict, correction, context, backend, seconds) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (utc_now(), transcript, user_text, reply, verdict, correction or None,
             json.dumps(context or [], ensure_ascii=False), backend, seconds))
        return int(cur.lastrowid)

def export_corrections(db=DB) -> list[dict]:
    \"\"\"Thumbs-down rows shaped like a training pair -- shaped like one, not
    saved as one. A person reviews these before any reaches the corpus.\"\"\"
    ...
""")

    D.h2(at, "G. Repository structure")
    D.table(at, ["Path", "Purpose"],
            [["app/stt.py", "Stage 1: language routing and transcription"],
             ["app/normalise.py", "Stage 2: transcription repair, with a self-test"],
             ["app/inference.py", "Stage 3: 4-bit base plus QLoRA adapter, generation"],
             ["app/feedback.py", "Stage 4: SQLite logging, review export"],
             ["app/streamlit_app.py", "Chat interface, voice and text"],
             ["app/pages/1_Admin.py", "Password-gated feedback review"],
             ["data/knowledge_base/", "Five domain YAML files, scope rules, system prompt"],
             ["data/generated/", "Corpus and the train/validation/test splits"],
             ["data/schema.sql", "Feedback table definition (the database is not committed)"],
             ["training/qlora_config.yaml", "Fine-tuning configuration"],
             ["training/to_llamafactory.py", "Corpus to training format, stamping the prompt"],
             ["training/colab_finetune.ipynb", "The training run"],
             ["training/colab_evaluate.ipynb", "The graded comparison of section 14"],
             ["eval/generate.py", "Runs a model over a split and records what it said"],
             ["eval/score.py", "Perplexity, BLEU, chrF, ROUGE-L, BERTScore"],
             ["eval/refusal_report.py", "Fallback accuracy and over-refusal"],
             ["eval/results/", "Committed result files behind every figure here"],
             ["archive/", "Stage 1 tooling that has run; nothing here is imported"]])
    D.caption(at, "Repository layout. The live application is five modules plus the "
                  "interface; everything else is data, evaluation or documentation.")


REFERENCES = [
    "Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient "
    "finetuning of quantized LLMs. Advances in Neural Information Processing Systems, 36.",
    "Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep "
    "bidirectional transformers for language understanding. Proceedings of NAACL-HLT 2019, "
    "4171-4186.",
    "Grattafiori, A., et al. (2024). The Llama 3 herd of models. arXiv:2407.21783.",
    "Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. "
    "(2021). LoRA: Low-rank adaptation of large language models. arXiv:2106.09685.",
    "Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. Text "
    "Summarization Branches Out, 74-81.",
    "Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). BLEU: A method for automatic "
    "evaluation of machine translation. Proceedings of ACL 2002, 311-318.",
    "Popovic, M. (2015). chrF: Character n-gram F-score for automatic MT evaluation. "
    "Proceedings of the Tenth Workshop on Statistical Machine Translation, 392-395.",
    "Post, M. (2018). A call for clarity in reporting BLEU scores. Proceedings of the Third "
    "Conference on Machine Translation, 186-191.",
    "Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: "
    "Evaluating text generation with BERT. International Conference on Learning "
    "Representations.",
    "Zheng, Y., Zhang, R., Zhang, J., Ye, Y., Luo, Z., Feng, Z., & Ma, Y. (2024). "
    "LLaMA-Factory: Unified efficient fine-tuning of 100+ language models. Proceedings of "
    "ACL 2024 (System Demonstrations).",
    "Zolkepli, H. (2018). Malaya: Natural language toolkit for the Malaysian language "
    "[Computer software]. https://github.com/malaysia-ai/malaya",
]


def references(D, e):
    at = D.at(lambda t: t.startswith("Add Part A references"))
    for ref in REFERENCES:
        D.para(at, ref)
    D.replace(lambda t: t.startswith("Add Part A references"),
              "References are listed alphabetically. Stage 1 sources appear above; the "
              "entries added here support Stages 2 to 4.")


def appendices(D, e):
    # -- Appendix A: fill the placeholder ---------------------------------
    D.replace(
        lambda t: t.startswith("Complete this honestly and specifically"),
        "The declaration below states what was generated, what was assisted, and what "
        "the team verified. It should be read and amended by both members before "
        "submission.")

    at = D.at(lambda t: t == "Appendix B: Prompt text used for decoder conditioning")

    D.h2(at, "Appendix A: Declaration of AI tool assistance (replaces the placeholder)")
    D.para(at, "Generated training data.", bold_lead=True)
    D.para(at, f"""The {e['pairs_total']} question-answer pairs in the training corpus were generated by a large language model, prompted with the team-written knowledge base, topic definitions and style constraints. The generation prompt is retained in the repository at archive/scripts/prompts/generate_pairs_prompt.md as the provenance record. The knowledge base itself, all {e['n_facts']} facts with their sources and stability markers, was written and verified by the team against agency portals; no fact was accepted from a model. Generated pairs were filtered automatically for duplication, sentence count and emoji, and a sample from every category was read by a team member.""")

    D.para(at, "Coding assistance.", bold_lead=True)
    D.para(at, """An AI coding assistant was used during implementation of the evaluation harnesses, the Colab notebooks, and the Stage 2 to 4 application modules, and in drafting sections of this report from the committed result files. All configuration values, evaluation results and reported figures were produced by running the code, not by a model asserting them; the result files behind every table are committed to the repository and can be regenerated.""")

    D.para(at, "Not generated.", bold_lead=True)
    D.para(at, """The speech test set, its 48 reference transcripts and the recordings themselves were authored and recorded by the team. The stratification design, the language-routing diagnosis, the choice of base model and the interpretation of every result in this report are the team's own.""")

    # -- new appendices, inserted before References is not possible (already
    #    passed), so append at the end of the document -----------------------
    D.h2_end("Appendix E: Transcription repair table")
    D.para_end(f"""The {len(e['repairs'])} entries applied by Stage 2. The rule for inclusion is that the token cannot legitimately occur in Malaysian rojak, either because Whisper invented it or because it is a spelling of a domain term the corpus never uses. Counts are occurrences in the 48-clip evaluation set.""")
    # Two kinds of entry, and the distinction is the justification: an invented
    # string is safe to repair because a correct transcript never contains it;
    # an orthographic variant is safe because the corpus only ever uses one form.
    ORTHOGRAPHIC = {"ayo", "efilling", "pasport", "license", "lho"}
    D.table_end(["Whisper wrote", "Repaired to", "Why it is safe to repair"],
                [[k, v, "Orthographic variant; the corpus uses one spelling"
                  if k in ORTHOGRAPHIC else "Invented token; cannot occur in a correct transcript"]
                 for k, v in e["repairs"].items()])
    D.caption_end("Transcription repairs applied before generation.")
    D.para_end("""The following high-frequency substitutions were deliberately excluded, because each is a real word being mistranslated and repairing it would corrupt the utterances where the word is correct.""")
    D.table_end(["Whisper wrote", "Came from", "Why it cannot be repaired"],
                [["saya", "can (9), do (6), i (6), balance (5), my (4)",
                  "Five different sources; no single repair is right"],
                 ["card", "roadtax (6), mykad (6)", "Ambiguous between two domains"],
                 ["buku", "pukul (8), tutup (2)", "buku is also the word for book"],
                 ["nampak", "nak (7)", "nampak is a common word in its own right"],
                 ["and", "n (7)", "Repairing would replace a correct word with an abbreviation"]])
    D.caption_end("Substitutions deliberately not repaired.")

    D.h2_end("Appendix F: System prompt")
    D.para_end("""Read from data/knowledge_base/_system_prompt.md by the training converter, the evaluation harness and the application, so that the text cannot drift between training, measurement and demonstration.""")
    D.code_end(e["system_prompt"])
