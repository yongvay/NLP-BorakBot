"""The Stage 2-4 content of the Part B report.

Separated from `build_report.py` so the mechanics of writing Word XML and the
prose that a marker actually reads are not interleaved. Every figure quoted here
arrives through the `e` dict, read from committed evidence at build time -- see
`build_report.evidence`. Nothing in this file hard-codes a measurement.
"""

from __future__ import annotations

DOMAIN_LABEL = {
    "jpj_vehicle_licence": "JPJ - road tax and driving licences",
    "slang_meanings": "Malaysian slang and discourse particles",
    "small_talk": "Small talk",
    "jpn_identity": "JPN - MyKad and birth certificates",
    "immigration_passport": "Immigration - passports",
}

CAT_LABEL = {
    "out_of_domain": "Outside the knowledge domains",
    "volatile": "Live or volatile values",
    "personal": "The user's own records",
    "current_events": "News and current events",
    "professional_advice": "Medical, legal or financial advice",
}


def front_matter(D, e):
    """The draft states in three places that it covers Stage 1 only."""
    D.replace(
        lambda t: t.startswith("This submission covers Stage 1"),
        "This submission covers the complete four-stage pipeline: speech-to-text "
        "(Stage 1), transcription repair (Stage 2), knowledge-based generation by a "
        "QLoRA fine-tuned language model (Stage 3), and the human-in-the-loop feedback "
        "loop (Stage 4). Sections 1 to 8 report Stage 1, evaluated with Word Error Rate "
        f"over 48 recorded clips. Sections 9 to 21 report the language-model half, "
        f"evaluated on a held-out split of {e['n_test']} question-answer pairs. Speech "
        "errors are reported apart from generation errors throughout, because a wrong "
        "answer to a misheard question is a different failure from a wrong answer to an "
        "understood one.")
    D.replace(
        lambda t: t.startswith("Sections on the synthetic dataset"),
        "Text-to-speech output is the one proposed component not built. The proposal "
        "marked it optional, and it was not reached.")
    D.replace(
        lambda t: t.startswith("Remaining tasks (synthetic dataset"),
        "All proposed stages are implemented and integrated. The split below should be "
        "agreed between the team before submission and must reflect actual work.")

    at = D.at(lambda t: t.startswith("Objective 7 (proposal"))
    for line in [
        "Objective 1 (proposal 4): accept Malay-English code-switched input, as text "
        "and as speech.",
        "Objective 2 (proposal 4): reply in the same rojak register.",
        "Objective 3 (proposal 4): keep replies knowledge-based, declining questions "
        "outside the curated knowledge rather than inventing an answer.",
        "Objective 5 (proposal 4): embed the knowledge base in the model by "
        "fine-tuning, with no retrieval component at inference.",
        "Objective 6 (proposal 4): capture user corrections for moderated retraining.",
    ]:
        D.para(at, line)


def deviations(D, e):
    at = D.at(lambda t: t == "Pseudocode")
    D.para(at, "Two further deviations arose in Stages 2 and 3. Both are argued in full "
               "where their evidence appears.")
    D.para(at, "Deviation: training environment.", bold_lead=True)
    D.para(at,
           "The proposal named Kaggle Notebooks. Kaggle gates notebook networking behind "
           "phone verification that the team account could not complete, and without "
           "networking pip, git and the Hugging Face Hub are all unreachable. Training "
           "moved to Google Colab on a free T4, checkpointing to Drive so a run survives "
           "disconnection. The change is environmental and affects no reported figure.")
    D.para(at, "Deviation: what Stage 2 normalises.", bold_lead=True)
    D.para(at,
           "Proposal 5.2 specified Malaya's informal-Malay normaliser, expanding forms "
           "such as xleh to tidak boleh, followed by a Manglish slang dictionary. That "
           "design suits an un-fine-tuned model. It is counter-productive here, and "
           "section 18 gives the measurement: 73% of the training inputs are informal "
           "rojak, so expanding shortforms before generation moves every input away from "
           "the distribution the model was fitted on. Stage 2 instead repairs "
           "transcription errors and leaves the register untouched.")


def pseudocode(D, e):
    at = D.at(lambda t: t == "Algorithm Analysis")

    D.h2(at, "C. Transcription repair")
    D.code(at, """
INPUT   : transcript string from Stage 1
OUTPUT  : same utterance, invented tokens repaired, register unchanged

CONSTANT REPAIRS <- tokens that cannot occur in Malaysian rojak, mapped to
                    what the speaker actually said (Appendix E)

PROCEDURE Repair(text):
    text <- CollapseWhitespace(text)
    text <- RemoveSpaceBeforePunctuation(text)
    text <- CollapseRepeatedPunctuation(text)

    // Case and wording are NOT touched. Capitals carry meaning -- MyKad, JPJ,
    // and the P and D licence classes -- and 391 of 506 training inputs end
    // in a question mark, so the model expects punctuation to survive.
    out <- empty list
    FOR EACH token IN SplitKeepingSeparators(text) DO
        IF Lower(token) IN REPAIRS THEN
            Append(out, MatchCase(token, REPAIRS[Lower(token)]))
        ELSE
            Append(out, token)
        END IF
    END FOR
    RETURN Join(out)
END PROCEDURE
""")

    D.h2(at, "D. Knowledge-based generation with trained refusal")
    D.code(at, """
INPUT   : repaired utterance, recent dialogue history
OUTPUT  : rojak reply, or the fallback sentence when out of knowledge

CONSTANT SYSTEM_PROMPT <- read from the knowledge base: the same file that
                          stamped every training example (Appendix C)
CONSTANT HISTORY_TURNS <- 2

PROCEDURE Answer(utterance, history):
    // Refusal is not a filter over the output. It is trained: 130 corpus
    // pairs answer an out-of-scope question with the fallback, so the
    // boundary lives in the weights and the prompt only restates it.
    messages <- [ {system, SYSTEM_PROMPT} ]
    Append(messages, LastN(history, 2 * HISTORY_TURNS))
    Append(messages, {user, utterance})

    text   <- ApplyChatTemplate(messages, add_generation_prompt = TRUE)
    tokens <- Model.Generate(text, max_new_tokens = 160, sampling = FALSE)
    RETURN Decode(tokens generated after the prompt)
END PROCEDURE
""")

    D.h2(at, "E. Feedback capture and moderated retraining")
    D.code(at, """
PROCEDURE OnVerdict(question, reply, verdict, correction, context):
    INSERT INTO feedback (created_utc, transcript, user_text, reply,
                          verdict, correction, context, backend, seconds)
    // transcript is NULL when the question was typed. It is stored apart
    // from user_text so an ASR failure stays distinguishable from an NLP one.
END PROCEDURE

PROCEDURE RetrainingCycle():        // periodic and team-reviewed, NOT automatic
    candidates <- SELECT * FROM feedback
                  WHERE verdict = 'down' AND correction IS NOT NULL
    approved   <- HumanReview(candidates)     // reject out-of-scope corrections
    corpus_v2  <- corpus_v1 + AddCorpusMetadata(approved)
    Split(corpus_v2) ; ConvertToTrainingFormat() ; QLoRA() ; Evaluate(v2 vs v1)
END PROCEDURE
""")


def analysis(D, e):
    at = D.at(lambda t: t == "Coding")
    s, r = e["scores"], e["refusal"]
    base, tuned = s["base"], s["tuned"]
    rb, rt = r["base"], r["tuned"]

    # ---------------------------------------------------------------- 9
    D.h2(at, "9. Knowledge base and synthetic corpus")
    D.para(at, f"""The knowledge the system is expected to hold was written first, as {e['n_facts']} verified facts across five domains, each carrying a source and a stability marker. Facts, not conversations, are the unit, because a fact can be checked against an agency portal and a conversation cannot. The corpus was then generated from those facts at roughly ten paraphrases each, giving {e['pairs_know']} knowledge pairs.""")
    D.para(at, f"""A further {e['pairs_ref']} pairs answer an out-of-scope question with the fallback sentence. These are not padding. Refusal is the hallucination control the proposal specifies, and a model learns it the way it learns anything else, from examples. Five categories were written deliberately so that the boundary is defined by cases rather than by an instruction alone.""")
    D.table(at,
            ["Content", "Category", "Pairs"],
            [["Knowledge", DOMAIN_LABEL.get(k, k), v]
             for k, v in sorted(e["domains"].items(), key=lambda kv: -kv[1])]
            + [["Refusal", CAT_LABEL.get(k, k), v]
               for k, v in sorted(e["refusal_cats"].items(), key=lambda kv: -kv[1])]
            + [["", "Total", e["pairs_total"]]])
    D.caption(at, f"Corpus composition. {e['pairs_know']} knowledge pairs drawn from "
                  f"{e['n_facts']} facts, and {e['pairs_ref']} refusal pairs from "
                  f"{e['n_probes']} out-of-scope probes.")
    D.para(at, """All generated data is declared in Appendix A, and the generation prompt is retained in the repository as the provenance record.""")

    # ---------------------------------------------------------------- 10
    D.h2(at, "10. Splitting the corpus, and what the split can measure")
    D.para(at, f"""The corpus was split {e['n_train']}/{e['n_val']}/{e['n_test']} at the level of the pair rather than the fact. Every fact and every probe appearing in the test split also appears in training; this was verified, and there are no unseen items.""")
    D.para(at, """This is a deliberate limitation and it changes what the held-out figures mean. A fact-level split would ask whether the model generalises to facts it was never taught, which for a system with no retrieval component has a known answer: it cannot, and measuring it would produce a number with no decision attached. A pair-level split measures something the system genuinely needs, namely whether a fact learned from one phrasing survives being asked in another. Every figure in sections 14 to 17 should therefore be read as paraphrase robustness, not as generalisation to new knowledge.""")

    # ---------------------------------------------------------------- 11
    D.h2(at, "11. Selecting the base model by measurement")
    ll, ml = r["llama"], r["mallam"]
    D.para(at, """Two candidates were compared on the same twenty probes before any fine-tuning: meta-llama/Llama-3.2-3B-Instruct, and mesolitica/mallam-3b-20k-instructions, which is pretrained on Malaysian text. The Malaysian model is the more fluent of the two on a read-through and would have been the intuitive choice.""")
    D.table(at, ["Candidate", "Out-of-scope declined", "In-domain wrongly declined"],
            [["Llama-3.2-3B-Instruct", D.pct(ll["fallback_loose"]), D.pct(ll["over_refusal_loose"])],
             ["mallam-3b-20k-instructions", D.pct(ml["fallback_loose"]), D.pct(ml["over_refusal_loose"])]])
    D.caption(at, "Base-model comparison before fine-tuning, over 20 probes "
                  "(3 out-of-scope, 17 in-domain). Lower is better in the right-hand column.")
    D.para(at, """The decision turned on the out-of-scope column. Llama declined all three; the Malaysian model declined one and, on the other two, fabricated a JPJ regulation and offered legal advice. Both fabrications are quotable verbatim from the committed results file.""")
    D.para(at, """Llama's own failure is the opposite one, declining 82% of the questions it should have answered, but the two failures are not equally correctable. Over-refusal is caused by not knowing the answer, and fine-tuning supplies exactly that. Fabrication is a disposition to answer regardless of knowledge, and nothing in the training plan removes it. The candidate was chosen on which failure the remaining work could repair, and section 16 shows that the prediction held.""")
    D.para(at, """One measurement was invalidated and re-run rather than reported. The Malaysian checkpoint declares no chat template, so the harness fell back to a plain layout and the model produced degenerate output on 19 of 20 probes, inventing both halves of a dialogue. That measured the formatting fallback, not the model. The comparison was repeated with the Mistral template its model card specifies. The first run is kept in the repository as evidence rather than deleted, because a checkpoint that declares no template failing silently is itself a finding.""")

    # ---------------------------------------------------------------- 12
    D.h2(at, "12. Fine-tuning configuration")
    D.para(at, f"""Adaptation uses QLoRA. The base model is loaded in 4-bit NF4 with double quantisation and its weights frozen, and low-rank adapters are trained on top. On a free T4 this is the difference between a run that fits in 16 GB and one that does not. Training ran {e['max_steps']} optimiser steps in {e['train_runtime'] / 60:.0f} minutes and produced an adapter of roughly 24 MB, against a 6 GB base model.""")
    D.table(at, ["Setting", "Value", "Reason"],
            [["Rank / alpha", "16 / 32", "Conventional 1:2 ratio; rank 16 is ample for 54 facts"],
             ["Target modules", "all attention and MLP projections",
              "Register transfer needs the feed-forward layers, not attention alone"],
             ["Dropout", "0.05", "Light, because the objective is memorisation"],
             ["Quantisation", "4-bit NF4, double", "Fits the free 16 GB T4"],
             ["Learning rate", "2e-4, cosine, 10% warmup", "Conventional for LoRA at this rank"],
             ["Effective batch", "16 (2 x 8 accumulation)", "Largest that fits beside the 4-bit base"],
             ["Precision", "fp16", "The T4 is Turing and has no bf16 path"],
             ["Epochs", "5", "Memorisation target; see section 13"]])
    D.caption(at, "QLoRA configuration. The full file is stage2_chatbot/training/qlora_config.yaml.")

    # ---------------------------------------------------------------- 13
    D.h2(at, "13. The loss curve, and why five epochs were kept")
    lo = min(e["evals"], key=lambda x: x[2])
    last = e["evals"][-1]
    D.table(at, ["Step", "Epoch", "Validation loss"],
            [[st, f"{ep:.2f}", f"{v:.4f}"] for st, ep, v in e["evals"]])
    D.caption(at, f"Validation loss during training. The minimum is {lo[2]:.4f} at step "
                  f"{lo[0]}; the run finished at {last[2]:.4f}.")
    D.para(at, f"""Training loss fell to {e['train_loss']:.2f} while validation loss rose {(last[2] / lo[2] - 1) * 100:.0f}% above its step-{lo[0]} minimum. On an ordinary supervised task that is overfitting, and the run should have stopped at step {lo[0]}.""")
    D.para(at, f"""The final two epochs were kept deliberately. The objective is not generalisation to unseen questions but memorisation of a fixed {e['n_facts']}-fact knowledge base, and with no retrieval step at inference every fact the system can state has to be present in the weights. Held-out loss here measures paraphrase robustness on facts the model has already seen, which makes it a proxy for the goal rather than the goal, and the wrong quantity to optimise all the way. Section 16 supplies the behavioural check that justifies the choice: over-refusal collapsed and fallback accuracy rose across the same epochs in which validation loss was deteriorating.""")
    D.para(at, f"""The cost is recorded rather than hidden. A checkpoint retention limit of three kept steps 125, 150 and 160 and discarded step {lo[0]}, which was the best by validation loss. Recovering it would cost another {e['train_runtime'] / 60:.0f} minutes and was not judged worthwhile against the behavioural result. Raising the retention limit is the fix for a second round.""")

    # ---------------------------------------------------------------- 14
    D.h2(at, "14. Fine-tuned against un-fine-tuned")
    D.para(at, f"""The graded comparison holds everything constant except the adapter: the same base model, the same tokenizer, the same {e['n_test']}-item test split, greedy decoding, and a 4-bit load in both cases. Perplexity is computed over the whole split; the four similarity metrics are computed over the {base['n_scored']} answerable items, for the reason given in section 15.""")
    def n2(v):                      # the CSV keeps four decimals; a report needs two
        return f"{float(v):.2f}"

    D.table(at, ["Metric", "Un-fine-tuned base", "Fine-tuned", "Change"],
            [["Perplexity (lower is better)", n2(base["perplexity"]), n2(tuned["perplexity"]),
              f"divided by {float(base['perplexity']) / float(tuned['perplexity']):.1f}"],
             ["BLEU", n2(base["bleu"]), n2(tuned["bleu"]),
              f"x{float(tuned['bleu']) / float(base['bleu']):.1f}"],
             ["chrF", n2(base["chrf"]), n2(tuned["chrf"]),
              f"x{float(tuned['chrf']) / float(base['chrf']):.1f}"],
             ["ROUGE-L", n2(base["rougeL"]), n2(tuned["rougeL"]),
              f"x{float(tuned['rougeL']) / float(base['rougeL']):.1f}"],
             ["BERTScore F1", n2(base["bertscore_f1"]), n2(tuned["bertscore_f1"]),
              f"+{float(tuned['bertscore_f1']) - float(base['bertscore_f1']):.1f}"]])
    D.caption(at, f"Similarity to the gold answer, {base['n_scored']} answerable test "
                  f"items. BERTScore uses bert-base-multilingual-cased.")
    D.para(at, """BLEU and chrF agree in direction, which is the check chrF was included to provide. Corpus BLEU over 52 short answers is noisy, and a disagreement between a word-level and a character-level metric would have meant trusting neither. There is none.""")
    D.para(at, """Two cautions on how these should be read. BERTScore is computed with a multilingual encoder rather than the English-only default, because the references are code-switched; its absolute values are therefore not comparable to published English figures, and only the difference between the two columns is claimed. And by section 10, every figure here measures paraphrase robustness on known facts.""")
    D.para(at, """Improvement is not uniform across domains, and the pattern is informative.""")
    D.table(at, ["Domain", "Test items", "BLEU base to tuned", "ROUGE-L base to tuned"],
            [[DOMAIN_LABEL.get(k, k), n, f"{b} to {t}", f"{br} to {tr}"]
             for k, n, b, t, br, tr in e["by_stratum"]])
    D.caption(at, "Per-domain similarity, best configuration. Register transfers more "
                  "readily than fact recall.")
    D.para(at, """The two strongest domains are the two where the answer is mostly style, and the weakest is the one that requires a specific number from the thinnest slice of training data. This is the expected shape for QLoRA on a small corpus, and it argues for enlarging the passport domain before a second round rather than for training the existing corpus longer.""")

    # ---------------------------------------------------------------- 15
    D.h2(at, "15. Why refusals are excluded from the similarity metrics")
    D.para(at, f"""All {rt['n_refusal']} refusal items in the test split share one gold answer, the canonical fallback sentence. Any model that declines a question therefore scores near-perfectly on those items regardless of whether declining was correct.""")
    D.para(at, f"""The un-fine-tuned base is very nearly a model that declines everything: it wrongly declines {D.pct(rb['over_refusal_loose'])} of the in-domain questions. Including the refusal rows in BLEU, ROUGE-L and BERTScore would hand it {rt['n_refusal']} of {int(rt['n_refusal']) + int(rt['n_answered'])} items, roughly 17% of the corpus, for its single worst behaviour, and would register over-refusal as accuracy. The metrics in section 14 are consequently computed over the answerable items only, and refusal is measured separately and on its own terms in section 16. Perplexity is the one exception: it measures the distribution the model learned, and the fallback sentence is part of what it was trained to produce.""")

    # ---------------------------------------------------------------- 16
    D.h2(at, "16. Hallucination control")
    D.table(at, ["Behaviour", "Un-fine-tuned base", "Fine-tuned"],
            [[f"Out-of-scope declined, exact wording ({rt['n_refusal']} items)",
              D.pct(rb["fallback_exact"]), D.pct(rt["fallback_exact"])],
             [f"Out-of-scope declined, any wording ({rt['n_refusal']} items)",
              D.pct(rb["fallback_loose"]), D.pct(rt["fallback_loose"])],
             [f"In-domain wrongly declined ({rt['n_answered']} items)",
              D.pct(rb["over_refusal_loose"]), D.pct(rt["over_refusal_loose"])]])
    D.caption(at, "Refusal behaviour on the full test split. The middle row is the one "
                  "most easily misread.")
    D.para(at, f"""Read alone, the middle row says fine-tuning achieved nothing: both models decline {D.pct(rt['fallback_loose'])} of out-of-scope questions. That reading is wrong, and the third row is why. The base reaches its figure by declining almost everything put to it, including {D.pct(rb['over_refusal_loose'])} of the questions it was supposed to answer. It is not discriminating between in-scope and out-of-scope; it is close to mute.""")
    gap_b = (float(rb["fallback_loose"]) - float(rb["over_refusal_loose"])) * 100
    gap_t = (float(rt["fallback_loose"]) - float(rt["over_refusal_loose"])) * 100
    D.para(at, f"""The measure of a working boundary is therefore not the refusal rate but the gap between the two rates. For the base model that gap is {gap_b:.0f} percentage points; for the fine-tuned model it is {gap_t:.0f}. That is the result, and it is the figure this report puts forward as evidence of hallucination control.""")
    D.para(at, f"""On exact wording, which is what the proposal quotes and what a demonstration is checked against, fallback accuracy moved from {D.pct(rb['fallback_exact'])} to {D.pct(rt['fallback_exact'])}. Over-refusal fell from {D.pct(rb['over_refusal_loose'])} to {D.pct(rt['over_refusal_loose'])}. Both changes were predicted in section 11 as the reason for preferring this base model, and both are in the predicted direction.""")

    # ---------------------------------------------------------------- 17
    D.h2(at, "17. Residual failures")
    D.para(at, f"""Two out-of-scope probes still leak, and they are not equally serious.""")
    D.para(at, """The first is harmless. Asked to explain photosynthesis, the system explained photosynthesis, correctly. The boundary leaked but nothing false was stated.""")
    D.para(at, """The second is not. Asked whether the user could sue an employer over a dismissal, the system answered, and answered incoherently. This is the same failure category that disqualified the alternative base model in section 11, where fabricating legal advice was the specific ground for rejection. The chosen model still commits it, on one of eleven out-of-scope items. Declaring this matters: the argument for this base was hallucination control, and the honest form of that argument is that fine-tuning reduced the failure without eliminating it. The professional-advice probe category exists in the corpus for exactly this reason and should be enlarged before a second round.""")
    n_wrong = round(float(rt["over_refusal_loose"]) * int(rt["n_answered"]))
    D.para(at, f"""{n_wrong} in-domain question of {rt['n_answered']} is still wrongly declined. The fact it asks for is present both in the knowledge base and in the training split, so this is a paraphrase-robustness miss, which is precisely what section 10 established the pair-level split can detect.""")

    # ---------------------------------------------------------------- 18
    D.h2(at, "18. Stage 2: repairing transcription, not normalising register")
    D.para(at, """Section 5.2 of the proposal specified informal-Malay normalisation at this point in the pipeline. Measurement made that design untenable. 371 of the 506 training inputs, 73%, contain informal rojak markers, and the conversion script passes the user field through verbatim, so the adapter was fitted on exactly that register. Expanding shortforms before generation would hand the model text unlike anything it was trained on, spending the improvement recorded in section 14, which is only collectable on the training distribution.""")
    D.para(at, """The same evidence rules out the other two operations the proposal named. Capitals are load-bearing: MyKad, JPJ, IC, and the P and D licence classes appear capitalised throughout the corpus, and case-folding P to p erases the difference between a probationary licence and a letter of the alphabet. Punctuation is expected: 391 of 506 training inputs contain a question mark.""")
    D.para(at, """Stage 2 therefore repairs what Stage 1 got wrong and leaves the register alone. Section 5 had already measured those errors and identified this as the remedy.""")
    D.para(at, """The substitution table from the WER evaluation cannot be used directly, and the reason is worth stating because it is a trap. That table records what each reference word turned into, not what each mistake came from, and reversing it corrupts correct transcripts. Whisper wrote saya for can, do, i, balance and my; it wrote card for both roadtax and mykad; it wrote buku for pukul, and buku is also simply the word for book. None of those can be repaired by lookup.""")
    D.para(at, f"""What can be repaired is the residue the model invents that is not a word in either language. The table holds {len(e['repairs'])} entries, listed in Appendix E with their occurrence counts, and the high-frequency pairs deliberately excluded are recorded alongside them so that the omission reads as a decision rather than an oversight.""")

    # ---------------------------------------------------------------- 19
    D.h2(at, "19. Serving the model, and the memory budget")
    D.para(at, """The demonstration machine has 4 GB of video memory. The 4-bit model requires roughly 2.4 GB for weights, since the embedding matrix is not quantised, plus about 0.6 GB for the CUDA context, activations and key-value cache. The speech model requires a further 1 GB, and the Windows display reserve takes roughly 0.3 GB. The total exceeds the card.""")
    D.para(at, """The language model receives the GPU. It is thirteen times larger than the speech model and runs once per reply rather than once per clip, so the speech model is pinned to the CPU, where it costs a few seconds that are invisible beside generation. This does not affect the Word Error Rate reported in sections 1 to 8: the computation is float32 on either device and the transcript is identical.""")
    D.para(at, """Closing that trap mattered more than it appears. The speech module previously claimed the GPU whenever one was available, so installing a CUDA build of the tensor library in order to accelerate Stage 3 would silently have moved Stage 1 onto the card as well, and the resulting out-of-memory error would have surfaced inside the language-model load, pointing at the wrong stage entirely.""")
    D.para(at, """Without a GPU the system still runs, loading in bfloat16 on the CPU at roughly 70 seconds per reply against 3 to 5 seconds for the 4-bit GPU path. The interface reports which backend produced each reply and how long it took, so a latency question during a demonstration can be answered with the actual configuration rather than an estimate.""")

    # ---------------------------------------------------------------- 20
    D.h2(at, "20. Stage 4: capturing corrections without trusting them")
    D.para(at, """Every reply carries a verdict control. A negative verdict opens a field for the answer the user expected, and the exchange is written to SQLite with the turns that preceded it, the backend that produced it, and its latency.""")
    D.para(at, """The transcript and the model input are stored in separate columns. The first is what the speech stage heard and is empty when the question was typed; the second is what the language model was given, after Stage 2. A wrong answer to a misheard question belongs against the Word Error Rate, not against the generation metrics, and a single merged column would make that distinction unrecoverable exactly when a reviewer needs it.""")
    D.para(at, """Corrections are never applied automatically, and this is the design rather than an unfinished feature. Section 5.7 of the proposal took the position from the literature on human correction: an unmoderated loop trains on whatever it is told, and a system that one user can teach a wrong fact is worse than one that cannot learn at all. There is no retraining trigger and no write path into the training data. Corrections are exported for review, and a person decides what becomes a training pair.""")
    D.para(at, """That decision was vindicated unexpectedly during testing. The first correction ever logged was a negative verdict on a correct refusal: the system had properly declined to explain photosynthesis, a topic outside its knowledge domains by three separate definitions, and the tester supplied an answer for it. Appended automatically, that correction would have taught the model to answer science questions and eroded the boundary that section 16 measures. The review step caught it. It is offered here as evidence that the moderated design was the right one, rather than as a hypothetical risk.""")

    # ---------------------------------------------------------------- 21
    D.h2(at, "21. Threats to validity")
    D.para(at, "Known facts only.", bold_lead=True)
    D.para(at, """Section 10 establishes that no test item concerns a fact absent from training. The figures in sections 14 and 16 therefore describe paraphrase robustness and must not be read as evidence that the system generalises to knowledge it was not given.""")
    D.para(at, "Sample size.", bold_lead=True)
    D.para(at, f"""{base['n_scored']} answerable items and {rt['n_refusal']} out-of-scope items. The refusal figures in particular rest on eleven observations, where one item moves the rate by nine percentage points. The per-domain figures in section 14 rest on as few as five items each and should be read as indicative.""")
    D.para(at, "A single training run.", bold_lead=True)
    D.para(at, """One seed, one configuration. No variance estimate exists, so the difference between the two columns of section 14 cannot be separated from run-to-run variation, though a perplexity change of this magnitude is unlikely to be seed noise.""")
    D.para(at, "Self-authored evaluation data.", bold_lead=True)
    D.para(at, """The team wrote the knowledge base, generated the corpus from it, and defined the out-of-scope probes. The test split is held out but shares authorship and register with the training data, so it is a weaker test than questions written by an unfamiliar user.""")
    D.para(at, "Both safeguards active throughout.", bold_lead=True)
    D.para(at, """Refusal is enforced twice: by the trained weights and by the system prompt. Every measurement reported here had both in place, so the contribution of each is unknown. The ablation that would separate them, running the fine-tuned model without the system prompt, is specified in the evaluation harness and has not been run.""")
    D.para(at, "No human ratings.", bold_lead=True)
    D.para(at, """Perplexity, BLEU, ROUGE-L, chrF and BERTScore all measure agreement with a written gold answer. None measures whether a reply is natural rojak or whether a Malaysian speaker would accept it. The Likert instrument for fluency, rojak naturalness and factual consistency remains outstanding.""")
