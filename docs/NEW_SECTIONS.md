# Results

*(Insert as a new top-level heading between **Algorithm Analysis** and **Coding**. Tables continue from 18, so nothing already in the report needs renumbering.)*

The evidence for every claim below is presented in full in the Algorithm Analysis; this section gathers the headline measurements into one place and states, for each project objective, whether it was achieved.

Two evaluations were run, and they are kept apart throughout. Stage 1 was measured by Word Error Rate over 48 purpose-recorded code-switched clips, 387 reference words. Stages 3 and 4 were measured on a held-out split of 63 question–answer pairs, against the same base model without the adapter. A wrong answer to a misheard question is a different failure from a wrong answer to an understood one, and merging the two would make neither diagnosable.

## Headline measurements

| Stage | Measure | Baseline | System as built | Detail |
| :---- | :---- | :---: | :---: | :---- |
| 1. Speech-to-text | Corpus WER, strict | 0.618 | **0.349** | Table 2 |
| 1. Speech-to-text | Corpus WER, lenient | 0.615 | **0.339** | Table 2 |
| 1. Speech-to-text | Best switch type (ms\_dom) | — | 0.155 | Table 6 |
| 1. Speech-to-text | Worst switch type (numeric) | — | 0.531 | Table 6 |
| 3. Generation | Perplexity | 23.75 | **3.65** | Table 11 |
| 3. Generation | BLEU | 1.28 | **19.82** | Table 11 |
| 3. Generation | chrF | 15.04 | **40.35** | Table 11 |
| 3. Generation | ROUGE-L | 8.11 | **39.63** | Table 11 |
| 3. Generation | BERTScore F1 | 64.28 | **78.29** | Table 11 |
| 3. Refusal | Out-of-scope declined, exact wording | 18% | **82%** | Table 13 |
| 3. Refusal | In-domain wrongly declined | 67% | **2%** | Table 13 |
| 3. Refusal | Discrimination gap between the two rates | 15 pts | **80 pts** | Table 13 |

*Table 19: Headline results. The Stage 1 baseline is the unassisted Malaysian checkpoint with no decoder prompt and no language routing; the Stage 3 baseline is the same base model, tokenizer, test split and decoding settings with the adapter removed.*

## Stage 1 — speech recognition

The best configuration reaches a corpus WER of 0.349 strict and 0.339 lenient, against 0.618 for the unassisted baseline: a 44% relative reduction. The improvement decomposes into three measured contributions. Supplying a short code-switched sample as decoder context is the largest single lever, worth 22 percentage points. The Malaysian fine-tuned checkpoint contributes a further 5 points over base Whisper at equal prompt conditions, and per-utterance language routing a further 5.

The proposal's target of WER ≤ 0.25 was **not met** on the full corpus. The report states this as measured rather than reframing it. An oracle analysis bounds what language routing alone can achieve on this test set at 0.277, so the remaining gap is not a routing problem: reaching 0.25 would additionally require number normalisation and Malaysian entity vocabulary. The target was set before any measurement was taken, and published WER for Malay–English code-switched ASR at this model size typically falls in the 0.30–0.50 range.

Error is not evenly distributed. Malay-dominant speech, discourse particles and true intra-sentential mixing are all at or below 0.234, while numeric and English-dominant utterances carry most of the error. The two are not equivalent failures: numeric error is largely a scoring artefact arising outside the system, whereas the English-dominant figure is a component of this system failing — the language detector hears English in all eight clips but is under-confident about it, assigning a median probability of 0.234 against a 0.5 threshold.

## Stage 3 — generation and hallucination control

Fine-tuning improved every similarity metric, and the two that were included as a cross-check on each other — BLEU at word level and chrF at character level — agree in direction, so neither is carrying the result alone. Perplexity fell by a factor of 6.5.

The hallucination-control result is the one that most repays careful reading, and Table 13 is easily misread. Both models decline roughly 82% of out-of-scope questions, which taken alone suggests fine-tuning achieved nothing. It did not: the un-fine-tuned base reaches that rate by declining almost everything put to it, including 67% of the questions it was supposed to answer. The measure of a working knowledge boundary is therefore not the refusal rate but the **gap** between the out-of-scope and in-domain refusal rates, and that gap moved from 15 percentage points to 80. Fallback accuracy on the exact required wording rose from 18% to 82%, and over-refusal fell from 67% to 2%.

Two out-of-scope probes still leak, and one of them matters: asked whether the user could sue an employer, the system answered rather than declining. This is the same failure category that disqualified the alternative base model during selection, so the honest form of the claim is that fine-tuning reduced the failure without eliminating it.

## Achievement against the project objectives

| # | Objective (proposal §4) | Achieved | Evidence |
| :---- | :---- | :---- | :---- |
| 1 | Accept Malay–English code-switched input, as text and as speech | Yes | Streamlit interface accepts typed rojak, in-browser recording and uploaded audio; Stage 1 evaluated at 0.349 WER (§1–8) |
| 2 | Reply in the same rojak register | Yes, by automatic metrics | BLEU 1.28 → 19.82, chrF 15.04 → 40.35, ROUGE-L 8.11 → 39.63 (§14). Not yet corroborated by human ratings — see Limitations |
| 3 | Keep replies knowledge-based, declining rather than inventing | Substantially | Discrimination gap 15 → 80 points; over-refusal 67% → 2% (§16). Two of eleven out-of-scope probes still leak (§17) |
| 4 | Integrate speech-to-text so users can converse by voice | Yes | Voice path runs end to end in the deployed application — in-browser recording or upload, transcription, then Stage 2 repair before generation. §19 records the CPU/GPU allocation that keeps it workable on a 4 GB card |
| 5 | Embed the knowledge base by fine-tuning, no retrieval at inference | Yes | 632-pair corpus from 54 verified facts; QLoRA adapter of ~24 MB over a frozen 4-bit base; perplexity 23.75 → 3.65 (§9, §12–14) |
| 6 | Capture user corrections for moderated retraining | Yes, as designed | Verdicts and corrections logged to SQLite with dialogue context; export for review with no automatic write path (§20) |
| 7 | Evaluate with appropriate quantitative metrics, speech errors reported apart from NLP errors | Yes | Two independent harnesses; WER reported over 48 clips, generation over 63 held-out pairs, never combined |

*Table 20: Project objectives and the evidence for each. The optional text-to-speech component named in the proposal was not built.*

Six of the seven objectives are met. Objective 3 is met substantially rather than completely, and the residual failures are reported in §17 rather than omitted.

---

# Discussion and Conclusion

*(Insert as a new top-level heading between **Coding** and **References**.)*

## What the findings mean

Four findings from this project generalise beyond it.

**In-domain decoder context matters more than model choice for code-switched ASR.** Supplying four lines of rojak as decoder context was worth 22 percentage points of WER — four times the benefit of swapping in a Malaysian-fine-tuned checkpoint. The prompt is not an instruction, and Whisper does not obey it; it shifts the decoder's language-model prior toward code-switched orthography where the audio is ambiguous. The composition of that prompt mattered as much as its presence: an initial, predominantly Malay prompt pushed English-dominant utterances into Malay translation, scoring 0.788 on that category. A prompt for code-switched speech must itself be code-switched.

**The bottleneck in code-switched ASR was language identification, not transcription.** The Malaysian checkpoint transcribes English competently when told to — 0.106 on English-dominant clips under a forced English token — but its own detector always lands on Malay, so those clips came back translated at 0.788. Separating identification from transcription across two models resolved most of it. What remains is a calibration limit: a 0.5 confidence threshold is a sensible default for monolingual audio and simply the wrong scale for utterances that are English sentences carrying Malay particles by design.

**A refusal rate is not a measure of a knowledge boundary.** This is the finding most likely to be misapplied elsewhere. A model that declines everything scores excellently on out-of-scope refusal and is useless. Only the gap between out-of-scope and in-domain refusal rates describes a boundary, and reporting the single rate would have credited the un-fine-tuned baseline for its worst behaviour. The same reasoning is why refusal items were excluded from BLEU, ROUGE-L and BERTScore: all eleven share one gold answer, so including them would have handed 17% of the corpus to a near-mute model.

**Validation loss was the wrong quantity to optimise to completion.** Validation loss reached its minimum at step 100 and rose 13% by the end of training, which on an ordinary supervised task is overfitting. The objective here was memorisation of a fixed 54-fact knowledge base with no retrieval at inference, so held-out loss measures paraphrase robustness on facts already seen — a proxy for the goal, not the goal. The behavioural check justifies the decision: over-refusal collapsed and fallback accuracy rose across exactly the epochs in which validation loss was deteriorating.

## Strengths of the solution

The evaluation is the strongest part of this work. Every reported figure is produced by committed code over committed data and can be regenerated; no number was hand-copied into the report, a rule adopted after three figures were found to have drifted from their source files. The speech test set is stratified by switch type with speakers balanced within each category rather than across the corpus, and that design is what made a speaker-dependent detection effect visible at all.

Design decisions were settled by measurement rather than intuition. The base model was chosen against a competitor that read as more fluent, on the ground that its failure mode — over-refusal from ignorance — was the one the remaining work could repair, while fabrication was not; §16 shows the prediction held. Stage 2 was rewritten away from the proposal's design because 73% of training inputs proved to be informal rojak, so expanding shortforms before generation would have moved every input off the distribution the adapter was fitted on.

The moderated feedback loop was vindicated in use rather than in principle. The first correction ever logged was a negative verdict on a *correct* refusal: the system properly declined to explain photosynthesis and the tester supplied an answer for it. Applied automatically, that single correction would have eroded the boundary the project spends most of its evaluation measuring.

The system is also honest about its own limits in operation: the interface reports which backend produced each reply and how long it took, and the speech model is deliberately pinned to the CPU so that Stage 3 keeps the 4 GB of video memory available.

## Limitations

**No human ratings were collected.** Perplexity, BLEU, chrF, ROUGE-L and BERTScore all measure agreement with a written gold answer. None measures whether a reply is natural rojak or whether a Malaysian speaker would accept it — which is precisely what Objective 2 claims. The Likert instrument for fluency, rojak naturalness and factual consistency is specified and its rating sheet built, but it was not administered. This is the single largest gap in the evaluation.

**The test split shares its knowledge with the training split.** The corpus was split at the level of the pair, not the fact, so every fact in the test split also appears in training. This is deliberate — a fact-level split would ask whether a model with no retrieval component can answer facts it was never taught, which has a known answer — but it means every generation figure describes paraphrase robustness and must not be read as generalisation to new knowledge.

**Small samples throughout.** 48 clips and 387 reference words for speech; 52 answerable and 11 out-of-scope items for generation, where one item moves the refusal rate by nine percentage points. Per-category and per-domain figures rest on as few as five items and are indicative only.

**Self-authored evaluation data.** The team wrote the knowledge base, generated the corpus from it, wrote the out-of-scope probes, and recorded the speech test set. The split is held out but shares authorship and register with the training data, so it is a weaker test than questions from an unfamiliar user. Both speakers are team members reading sentences they wrote; real users will be less fluent readers, and the recordings are read speech in quiet rooms rather than spontaneous conversation with disfluencies and background noise.

**A single training run and two entangled safeguards.** One seed, one configuration, so no variance estimate exists. Refusal is enforced twice — by the trained weights and by the system prompt — and both were active in every measurement, so the contribution of each is unknown. The ablation that would separate them is specified in the harness and was not run.

**Scope and coverage.** The knowledge base was narrowed twice, and two of its five domains remain both thin and narrow: all five JPN facts concern a lost MyKad and all five Immigration facts concern renewal. Someone asking about a first MyKad hits the fallback on a question squarely inside the stated domain, which looks like the boundary failing rather than working. The optional text-to-speech component was not built.

## Future enhancements

In the order we would take them:

1. **Administer the Likert instrument.** Both members rate a blinded sample of fine-tuned and base outputs for fluency, rojak naturalness and factual consistency, and report inter-rater agreement. The blinded rating-sheet generator and its scorer already exist in the evaluation harness; this is the cheapest large improvement available, and it closes the gap under Objective 2.
2. **Run the second retraining round.** Corrections are being logged and the export path exists, but round 2 has not been run. It is what makes the human-in-the-loop claim demonstrable rather than architectural.
3. **Calibrate the language router rather than tune its threshold.** The sweep finds a minimum of 0.277 at a threshold of 0.05, but that optimum sits beside a cliff — one grid step further down, WER is 1.274 — and choosing it on the same 48 clips it is evaluated on would be leakage. The defensible fix is a detector calibrated on code-switched audio, or a short second-pass decision using both language branches.
4. **Broaden the two thin domains** by roughly twenty facts each, favouring breadth over depth, and regenerate those domains rather than training the existing corpus longer. Per-domain results show register transfers readily while fact recall is limited by the thinnest slice of training data.
5. **Enlarge the professional-advice probe category** and re-measure. One leak in eleven is the residual failure with real-world consequences.
6. **Raise the checkpoint retention limit** so the best-by-validation checkpoint is not discarded, and run a second seed to obtain a variance estimate.
7. **Record additional speakers.** The English-detection effect is currently confounded with one speaker's accent; three or four more voices would establish whether it is a property of the detector or of the corpus.
8. **Add text-to-speech**, the one proposed component not built, to complete the conversational loop.

## Conclusion

BorakBot accepts Bahasa Rojak as speech or text, transcribes it with a language-routed Malaysian Whisper pipeline, repairs the errors that transcription introduces without disturbing the register, answers from a knowledge base embedded in the model's weights, declines what it was not taught, and logs corrections for moderated retraining. All four proposed stages are implemented, integrated and evaluated; six of seven objectives are met and the seventh substantially.

Two headline numbers describe the outcome. Speech recognition improved 44% relative to an unassisted baseline, to a WER of 0.349 — short of the 0.25 the proposal targeted, and reported as measured, with an oracle analysis showing where the remaining error actually lives. Generation improved on every similarity metric, most consequentially in hallucination control, where the gap between out-of-scope and in-domain refusal widened from 15 percentage points to 80.

The project's more transferable contribution is methodological. Several results here would have been reported wrongly under the obvious reading — the refusal rate that credits a near-mute model, the validation curve that says stop at step 100, the substitution table that cannot be reversed into a repair table, the threshold optimum sitting beside a cliff — and in each case the correct interpretation came from measuring the thing itself rather than a convenient proxy for it. The clearest evidence that this discipline was worth adopting is the first correction the system ever logged, which was wrong, and which the moderated design caught.
