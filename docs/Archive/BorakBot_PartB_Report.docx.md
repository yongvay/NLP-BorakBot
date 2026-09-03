**TUNKU ABDUL RAHMAN UNIVERSITY OF MANAGEMENT AND TECHNOLOGY**

FACULTY OF COMPUTING AND INFORMATION TECHNOLOGY

**BMDS2123 NATURAL LANGUAGE PROCESSING 202605**

Assignment Title: **BorakBot — A Knowledge-Based Bahasa Rojak Chatbot with Speech Input**

Programme                       : \[ programme name, e.g. Bachelor of Data Science \]

Tutorial Group               : **4**

Lecturer/Tutor's Name  : \[ tutor name \]

Date of submission      : **Week 12 (Monday – 31 August 2026, before 11.59 pm)**

**Team Members:**

| No | Student Name | Student ID | Signature | Marks (For Lecturer / Tutor use) |
| :---- | :---- | :---- | :---- | :---- |
| 1\. | Ng Yong Vay | 25WMR09608 |  |  |
| 2\. | Teo Pei Qi | 25WMR09625 |  |  |

# **I. Student Declaration of Originality**

We confirm that we have read and shall comply with all the terms and conditions of TAR University of Management and Technology plagiarism policy.

We declare that this assignment is free from all forms of plagiarism and for all intents and purposes is our own properly derived work.

| Name: | Ng Yong Vay | Teo Pei Qi |
| :---- | :---- | :---- |
| Photo: |  |  |
| Signature: |  |  |

# **II. Module and Task Handled by Each Team Member and Contribution**

All proposed stages are implemented and integrated. The split below should be agreed between the team before submission and must reflect actual work.

| No. | Team Member | Task(s) Completed | Contribution (%) |
| :---- | :---- | :---- | :---- |
| 1\. | Ng Yong Vay | Speech test-set design and stratification; audio conversion pipeline; WER evaluation harness; language-routing design, diagnosis and threshold sensitivity analysis; JPJ knowledge-base domain; corpus generation prompt and validator; QLoRA configuration and training run; similarity evaluation harness (perplexity, BLEU, chrF, ROUGE-L, BERTScore); error analysis and reporting. | 50% |
| 2\. | Teo Pei Qi | Bahasa Rojak utterance authoring and validation; speech recording; orthography map construction from substitution analysis; results verification; JPN, Immigration, slang and small-talk knowledge-base domains; out-of-scope probe design and corpus review; Stage 2 transcription-repair table and module; Streamlit interface and Stage 4 feedback loop; refusal and hallucination-control evaluation; reporting. | 50% |

# **Table of Contents**

*(In Word: right-click → Update Field to populate page numbers.)*

# **Introduction**

BorakBot is a knowledge-based conversational agent that accepts Bahasa Rojak — Malay-English code-switched speech and text — and replies in the same informal register. The system comprises four stages: speech-to-text capture, rojak normalisation, response generation by a QLoRA fine-tuned large language model, and a human-in-the-loop correction loop. This report documents the design, implementation and evaluation of all four stages.

## **Scope of this report**

This submission covers the complete four-stage pipeline: speech-to-text (Stage 1), transcription repair (Stage 2), knowledge-based generation by a QLoRA fine-tuned language model (Stage 3), and the human-in-the-loop feedback loop (Stage 4). Sections 1 to 8 report Stage 1, evaluated with Word Error Rate over 48 recorded clips. Sections 9 to 21 report the language-model half, evaluated on a held-out split of 63 question-answer pairs. Speech errors are reported apart from generation errors throughout, because a wrong answer to a misheard question is a different failure from a wrong answer to an understood one.

Text-to-speech output is the one proposed component not built. The proposal marked it optional, and it was not reached.

## **Objectives addressed**

* Objective 1 (proposal §4): accept Malay-English code-switched input, as text and as speech.

* Objective 2 (proposal §4): reply in the same rojak register.

* Objective 3 (proposal §4): keep replies knowledge-based, declining questions outside the curated knowledge rather than inventing an answer.

* Objective 4 (proposal §4): integrate speech-to-text so users can converse by voice.

* Objective 5 (proposal §4): embed the knowledge base in the model by fine-tuning, with no retrieval component at inference.

* Objective 6 (proposal §4): capture user corrections for moderated retraining.

* Objective 7 (proposal §4): evaluate the system with appropriate quantitative metrics — here, Word Error Rate (WER) on code-switched speech, reported separately from downstream NLP errors.

## **Deviation from the Part A proposal**

Proposal §5.6 selected OpenAI Whisper (Radford et al., 2023\) and identified its central weakness: Whisper assigns a single language token per 30-second segment, so intra-sentential code-switching risks being rendered entirely in one language's orthography. Our implementation retains Whisper but strengthens it in two ways not described in Part A:

* Transcription uses **mesolitica/malaysian-whisper-small-v3**, a fine-tune of openai/whisper-small on Malaysian speech including Manglish. It is the same architecture and parameter count (244M) as the base model, with different weights.

* A language-routing step selects the language token per utterance rather than accepting the model's default.

Both changes address the exact limitation §5.6 predicted, and both are justified empirically in the Algorithm Analysis below. Mesolitica was already cited in Part A as a candidate base model provider (MaLLaM), so this is consistent with the proposal's stated direction.

Two further deviations arose in Stages 2 and 3\. Both are argued in full where their evidence appears.

**Deviation: training environment.** The proposal named Kaggle Notebooks. Kaggle gates notebook networking behind phone verification that the team account could not complete, and without networking pip, git and the Hugging Face Hub are all unreachable. Training moved to Google Colab on a free T4, checkpointing to Drive so a run survives disconnection. The change is environmental and affects no reported figure.

**Deviation: what Stage 2 normalises.** Proposal §5.2 specified Malaya's informal-Malay normaliser, expanding forms such as xleh to tidak boleh, followed by a Manglish slang dictionary. That design suits an un-fine-tuned model. It is counter-productive here, and section 18 gives the measurement: 73% of the training inputs are informal rojak, so expanding shortforms before generation moves every input away from the distribution the model was fitted on. Stage 2 instead repairs transcription errors and leaves the register untouched.

# **Pseudocode**

## **A. Speech-to-text with language routing**

INPUT   : audio file (16 kHz mono WAV)  
OUTPUT  : transcript string in the language actually spoken

CONSTANT ROJAK\_PROMPT \<- short code-switched sample text, containing both  
                         Malay and English words in their own orthography

PROCEDURE Transcribe(audio):  
    // Stage 1a — language identification  
    mel        \<- LogMelSpectrogram(PadOrTrim(audio, 30 seconds))  
    probs      \<- VanillaWhisper.DetectLanguage(mel)  
    p\_en       \<- probs\['en'\]          // recorded for sensitivity analysis  
    IF Argmax(probs) \= 'en' AND p\_en \> EN\_THRESHOLD THEN  
            lang \<- 'en'  
    ELSE  
            lang \<- 'ms'        // asymmetric: a wrong 'en' is rare but unbounded  
    END IF

    // Stage 1b — transcription  
    prompt\_ids \<- Tokenise(ROJAK\_PROMPT)  
    tokens     \<- MalaysianWhisper.Generate(  
                      features \= Encode(audio),  
                      language \= lang,  
                      task     \= 'transcribe',  
                      context  \= prompt\_ids)  
    text       \<- Decode(tokens, skip\_special\_tokens \= TRUE)  
    RETURN StripEchoedPrompt(text, ROJAK\_PROMPT)  
END PROCEDURE

## **B. Evaluation procedure**

INPUT   : manifest of (audio\_path, reference, switch\_type, speaker)  
OUTPUT  : corpus WER per configuration and per switch type

PROCEDURE Evaluate(manifest, configurations):  
    FOR EACH config IN configurations DO  
        FOR EACH clip IN manifest DO  
            hypothesis \<- Transcribe(clip.audio, config)  
            Store(config, clip.id, hypothesis)  
        END FOR  
    END FOR

    FOR EACH stored (reference, hypothesis) DO  
        r \<- Normalise(reference)     // lowercase, strip punctuation, collapse spaces  
        h \<- Normalise(hypothesis)  
        (S, D, I, N) \<- Align(r, h)   // substitutions, deletions, insertions, ref words  
        Accumulate(errors \+= S \+ D \+ I, words \+= N)  
    END FOR

    // Corpus WER, NOT the mean of per-clip rates: averaging per-clip WER  
    // over-weights short utterances.  
    RETURN errors / words  
END PROCEDURE

## **C. Transcription repair**

INPUT   : transcript string from Stage 1  
OUTPUT  : same utterance, invented tokens repaired, register unchanged

CONSTANT REPAIRS \<- tokens that cannot occur in Malaysian rojak, mapped to  
                    what the speaker actually said (Appendix E)

PROCEDURE Repair(text):  
    text \<- CollapseWhitespace(text)  
    text \<- RemoveSpaceBeforePunctuation(text)  
    text \<- CollapseRepeatedPunctuation(text)

    // Case and wording are NOT touched. Capitals carry meaning \-- MyKad, JPJ,  
    // and the P and D licence classes \-- and 391 of 506 training inputs end  
    // in a question mark, so the model expects punctuation to survive.  
    out \<- empty list  
    FOR EACH token IN SplitKeepingSeparators(text) DO  
        IF Lower(token) IN REPAIRS THEN  
            Append(out, MatchCase(token, REPAIRS\[Lower(token)\]))  
        ELSE  
            Append(out, token)  
        END IF  
    END FOR  
    RETURN Join(out)  
END PROCEDURE

## **D. Knowledge-based generation with trained refusal**

INPUT   : repaired utterance, recent dialogue history  
OUTPUT  : rojak reply, or the fallback sentence when out of knowledge

CONSTANT SYSTEM\_PROMPT \<- read from the knowledge base: the same file that  
                          stamped every training example (Appendix F)  
CONSTANT HISTORY\_TURNS \<- 2

PROCEDURE Answer(utterance, history):  
    // Refusal is not a filter over the output. It is trained: 130 corpus  
    // pairs answer an out-of-scope question with the fallback, so the  
    // boundary lives in the weights and the prompt only restates it.  
    messages \<- \[ {system, SYSTEM\_PROMPT} \]  
    Append(messages, LastN(history, 2 \* HISTORY\_TURNS))  
    Append(messages, {user, utterance})

    text   \<- ApplyChatTemplate(messages, add\_generation\_prompt \= TRUE)  
    tokens \<- Model.Generate(text, max\_new\_tokens \= 160, sampling \= FALSE)  
    RETURN Decode(tokens generated after the prompt)  
END PROCEDURE

## **E. Feedback capture and moderated retraining**

PROCEDURE OnVerdict(question, reply, verdict, correction, context):  
    INSERT INTO feedback (created\_utc, transcript, user\_text, reply,  
                          verdict, correction, context, backend, seconds)  
    // transcript is NULL when the question was typed. It is stored apart  
    // from user\_text so an ASR failure stays distinguishable from an NLP one.  
END PROCEDURE

PROCEDURE RetrainingCycle():        // periodic and team-reviewed, NOT automatic  
    candidates \<- SELECT \* FROM feedback  
                  WHERE verdict \= 'down' AND correction IS NOT NULL  
    approved   \<- HumanReview(candidates)     // reject out-of-scope corrections  
    corpus\_v2  \<- corpus\_v1 \+ AddCorpusMetadata(approved)  
    Split(corpus\_v2) ; ConvertToTrainingFormat() ; QLoRA() ; Evaluate(v2 vs v1)  
END PROCEDURE

# **Algorithm Analysis**

## **1\. Test set construction**

No public Bahasa Rojak speech corpus exists, so a purpose-built evaluation set was constructed. Forty-eight utterances were written across six categories, eight per category, and recorded by both team members. Reference transcripts were written before recording; transcribing our own audio afterwards would have biased the references toward whatever was actually said.

The split between speakers was made within each category — four utterances each — rather than by dividing the list in half. A straight split would have confounded speaker with category, making it impossible to tell whether a category scored badly because of code-switching difficulty or because of one speaker's voice.

| Category | n | What it tests |
| :---- | :---- | :---- |
| ms\_dom | 8 | Malay matrix sentence with an English insertion |
| en\_dom | 8 | English matrix sentence with Malay discourse particles |
| balanced | 8 | True intra-sentential code-switching |
| particles | 8 | Discourse particles: lah, lor, meh, kan, kot |
| numeric | 8 | Prices, percentages and dates (4 Malay numerals, 4 English) |
| entity | 8 | Malaysian named entities: MyJPJ, KWSP, LHDN, TNB |

*Table 1: Stratification of the speech test set (48 utterances, 387 reference words)*

Numeric references are written as spoken ("empat puluh lima ringgit", not "RM45.90"). Writing digits while speaking words would score a perfect transcription as several errors, since WER compares surface strings.

## **2\. Evaluation metric and normalisation policy**

Word Error Rate is reported as total errors divided by total reference words across the corpus, not as the mean of per-clip rates. Averaging per-clip WER over-weights short utterances: one error in a four-word clip would count as heavily as four errors in a sixteen-word clip.

Both reference and hypothesis are lowercased, stripped of punctuation, and whitespace-collapsed before alignment. Beyond that, a decision was required. Whisper may write "cek" where the reference says "check" — the same word in the other language's orthography. Two scores are therefore reported:

* **Strict** — counts orthographic variants as errors. Justified because the downstream normaliser and fine-tuned LLM receive the surface string, so a wrong spelling is a real downstream problem.

* **Lenient** — canonicalises a small, documented set of same-word spelling variants first (cek/check, akaun/account, klinik/clinic).

Translations are deliberately excluded from the lenient map. When "how" becomes "bagaimana" the meaning survives but the transcript is still wrong for Stage 2, which expects informal rojak rather than formal Malay. The gap between strict and lenient quantifies how much of the error is orthographic rather than acoustic.

## **3\. Configurations compared**

Nine configurations were evaluated over two factors — model and prompt — plus the language token, which emerged as the dominant variable during analysis.

| Configuration | WER (strict) | WER (lenient) | S | I | D |
| :---- | :---: | :---: | :---: | :---: | :---: |
| malaysian\_prompt\_routed | 0.349 | 0.339 | 90 | 11 | 34 |
| malaysian\_prompt\_auto | 0.398 | 0.390 | 105 | 12 | 37 |
| malaysian\_prompt (forced ms) | 0.398 | 0.390 | 105 | 12 | 37 |
| vanilla\_prompt\_routed | 0.429 | 0.419 | 124 | 14 | 28 |
| vanilla\_prompt | 0.447 | 0.439 | 108 | 15 | 50 |
| vanilla\_noprompt | 0.592 | 0.581 | 166 | 8 | 55 |
| malaysian\_noprompt\_auto | 0.600 | 0.597 | 175 | 13 | 44 |
| malaysian\_noprompt (forced ms) | 0.618 | 0.615 | 181 | 13 | 45 |
| malaysian\_prompt\_en (diagnostic) | 1.274 | 1.264 | 90 | 379 | 24 |

*Table 2: WER by configuration, 48 clips, 387 reference words. S/I/D \= substitutions, insertions, deletions.*

Three findings follow directly from Table 2\.

**Prompting is the single largest lever.** Supplying a short code-switched sample as decoder context lowers WER substantially on both models: 22 percentage points on the Malaysian checkpoint (0.618 → 0.398) and 14 on the base model (0.592 → 0.447). Whisper's decoder is simultaneously an acoustic model and a language model; where audio is ambiguous, the language-model prior breaks the tie, and in-domain context shifts that prior toward rojak orthography. It is not an instruction — the model will not obey commands placed in the prompt.

**Prompt composition matters as much as its presence.** An initial, predominantly Malay prompt pushed the decoder so far toward Malay that English-dominant utterances were returned translated, scoring 0.788 on that category. Rewriting the prompt with English words in English orthography corrected this. A prompt for code-switched speech must itself be code-switched.

**The Malaysian fine-tune outperforms the base model.** At equal prompt conditions, 0.398 against 0.447. The advantage is consistent rather than marginal, and is expected given the checkpoint was distilled on Malaysian speech including Manglish.

## **4\. Language selection: diagnosis and resolution**

Whisper's decoder begins with the tokens \<|startoftranscript|\> \<|LANG|\> \<|transcribe|\>. The middle token does not describe the audio; it specifies the language the model should write in. When it matches the spoken language the result is transcription. When it does not, the model produces the audio's content in the wrong language — that is, translation.

Initial runs forced Malay, and en\_dom scored 0.788. Inspecting transcripts showed why:

  reference : where can i find cheap flight ticket meh  
  hypothesis: di mana saya boleh mencari tiket penerbangan yang murah  
The meaning is preserved but every word differs, giving WER \= 1.12 (8 substitutions \+ 1 insertion over 8 reference words; WER exceeds 1.0 because insertions are not bounded by reference length). This is a genuine failure for BorakBot, not merely a scoring artefact: the discourse particle "meh" is lost, and Stage 2's normaliser and fine-tuned model expect informal rojak, not formal Malay.

Setting language=None did not help — results were byte-identical to forced Malay, because the Mesolitica checkpoint pins the Malay token in its generation configuration. A diagnostic run forcing English isolated the cause:

| Category | Forced Malay | Forced English |
| :---- | :---: | :---: |
| en\_dom | 0.788 | 0.106 |
| ms\_dom | 0.155 | 6.483 |

*Table 3: Forced-language diagnostic. The model transcribes English well when instructed to; forcing the wrong language causes runaway repetition.*

The checkpoint therefore transcribes English competently — its language detector is the defect. The resolution is a two-model pipeline: base Whisper performs language identification (one encoder pass, unbiased), and the Malaysian model transcribes using that token. Routing reduced overall WER from 0.398 to 0.349.

**The threshold is deliberately asymmetric,** requiring confident English before it is selected, because the two errors are not equally expensive. Per-clip inspection shows the asymmetry is narrower than the aggregate figure suggests. Under forced English, seven of the eight ms\_dom clips transcribe correctly or near-correctly (0, 0, 0, 0, 1, 1 and 5 errors), and the entire 6.483 category figure is produced by one clip, s05\_pq, which enters runaway repetition and accumulates 362 insertions against a seven-word reference. Forcing the wrong language is therefore not uniformly catastrophic; it is usually harmless and occasionally unbounded. A low-probability, unbounded loss still justifies caution — WER has no upper limit, so a single such clip can dominate a corpus of this size — but the justification is expected cost under a heavy tail, not uniform damage.

## 5\. Threshold sensitivity of the routing decision

The routing decision was originally logged only as its outcome, 'en' or 'ms'. That record cannot distinguish two situations with opposite implications: a clip whose English is heard but scored just below the cutoff, which would mean the constant is mis-set, from a clip whose English is not heard at all, which would mean routing has reached its ceiling. The detector's English probability is therefore now recorded per clip (run\_wer.py \--detect-only, writing detector\_probs.csv).

| Category | p\_en median | p\_en range | Routed to English |
| :---- | :---- | :---- | :---- |
| en\_dom | 0.234 | 0.067 – 0.671 | 3 of 8 |
| particles | 0.011 | 0.000 – 0.466 | 0 of 8 |
| numeric | 0.002 | 0.000 – 0.524 | 1 of 8 |
| ms\_dom | 0.004 | 0.000 – 0.084 | 0 of 8 |
| balanced | 0.007 | 0.000 – 0.048 | 0 of 8 |
| entity | 0.005 | 0.000 – 0.027 | 0 of 8 |

*Table 4: English probability assigned by the vanilla Whisper detector, by switch type. Threshold for selecting English is 0.5.*

The detector is not deaf to the English; it is under-confident about it. Every en\_dom clip receives a non-trivial English probability (minimum 0.067), but the category median is 0.234 — less than half the threshold. These utterances are English matrix sentences carrying Malay discourse particles by design, so the detector sees a mixture and hedges. A cutoff of 0.5 is a reasonable default for monolingual audio and is simply the wrong scale for code-switched audio.

The distributions nevertheless overlap, so no cutoff separates the categories cleanly. en\_dom spans 0.067 to 0.671, while one numeric clip reaches 0.524 and one particles clip 0.466. Any threshold low enough to capture all eight English-dominant clips also captures clips from categories where English is the wrong choice. Routing on this signal has a floor that is a property of the detector, not of the constant.

Sensitivity was measured without re-transcribing anything. Routing selects between two language tokens, and both branches already exist for every clip as the malaysian\_prompt and malaysian\_prompt\_en configurations, so the routed transcript at any threshold is a selection over cached text. The procedure reconstructs the measured 0.349 exactly at the threshold in use, which is what establishes that the reconstruction is faithful before any other row is read.

| Threshold | Clips routed to English | WER (strict) | WER (lenient) |
| :---- | :---- | :---- | :---- |
| 0.00 | 48 | 1.2739 | 1.2636 |
| 0.05 | 12 | 0.2765 | 0.2661 |
| 0.10 | 9 | 0.2997 | 0.2894 |
| 0.25 | 6 | 0.3256 | 0.3152 |
| 0.30 | 5 | 0.3488 | 0.3385 |
| 0.55 | 3 | 0.3540 | 0.3437 |
| 0.60 | 2 | 0.3669 | 0.3566 |
| 0.70 | 0 | 0.3979 | 0.3902 |

*Table 5: Routed WER as a function of the English-detection threshold. Rows are the points at which the result changes; the full 21-point curve is in archive/eval/speech/results/threshold\_sweep.csv.*

The lowest WER on this corpus is 0.2765 at a threshold of 0.05, against 0.3488 at the threshold actually used — a headroom of 0.072. That figure coincides, to four decimal places, with the oracle bound reported in Section 7, which was obtained by a different method; two independent routes to the same value is a useful consistency check on both.

**Despite that headroom,** the threshold was not moved, for two reasons. The first is methodological: selecting a decision rule by its score on the 48 clips it is then evaluated on is leakage, and the corpus is too small to hold out a tuning split. It is the same objection that prevented Malaysian entity names being added to the decoder prompt. The second is that the minimum is unstable. At a threshold of 0.05 one ms\_dom clip (s07\_pq, p\_en \= 0.084) crosses into English and happens to be transcribed correctly either way; the clip that does cause runaway repetition, s05\_pq, sits far below and is not selected. The favourable result depends on which Malay clip happens to carry the higher probability. One step further down the grid, at a threshold of 0.00, the whole corpus is forced to English and WER rises to 1.274. The optimum on this data sits immediately beside a cliff, and a rule chosen there would not be expected to transfer to new speakers.

The reported configuration therefore retains the threshold chosen before measurement. Table 5 is presented as a sensitivity analysis: it quantifies what the caution costs on this corpus, and it makes the value of the constant an evidenced choice rather than an assumed one.

## **6\. Error analysis by switch type**

| Category | Reference words | WER (strict) |
| :---- | :---: | :---: |
| ms\_dom | 58 | 0.155 |
| particles | 58 | 0.190 |
| balanced | 64 | 0.234 |
| entity | 60 | 0.367 |
| en\_dom | 66 | 0.530 |
| numeric | 81 | 0.531 |
| **Overall** | **387** | **0.349** |

*Table 6: WER by switch type for the best configuration (malaysian\_prompt\_routed)*

**The §5.6 prediction was not confirmed.** Proposal §5.6 predicted that intra-sentential switching would be the hardest case. In practice 'balanced' scored 0.234 — third best, and below the 0.25 target. The prediction was reasonable but our data does not support it: the prompt appears to supply enough code-switched context for the decoder to handle mixed utterances, whereas the failures cluster elsewhere.

**Numeric is the weakest category (0.531),** and the cause is largely representational rather than acoustic. References are written as spoken while Whisper writes digits: 'thirty one' becomes '31', 'twenty twenty six' becomes '2026'. A word-level orthography map cannot express these many-to-one conversions; a dedicated number normaliser would be required. The figure is reported as measured, with the caveat stated.

**Entity errors (0.367) are genuine recognition failures.** 'MyJPJ' is transcribed 'jbj'. Malaysian agency names are rare in the training distribution. Supplying them as prompt vocabulary is a known remedy, but since these entities appear in the test set it would constitute leakage, so it was not applied.

**en\_dom (0.530)** is a routing failure rather than a transcription failure. Five of its eight clips are routed to Malay and returned as Malay prose. Under forced English the same category scores 0.106, so the model can write these utterances correctly when it is told which language to use. Section 5 identifies the mechanism: the detector assigns these clips a median English probability of 0.234, below the 0.5 threshold, while still detecting English in all eight. The remedy is a better-calibrated detector, not a better transcriber.

**Excluding the two weakest categories, the remaining 240 reference words (62% of the corpus) score 0.238 — within the proposal's ≤0.25 target. The two exclusions are not equally defensible and should not be read as equivalent: numeric is a scoring artefact arising outside the system, whereas en\_dom is a failure of a component of this system. The 0.349 figure over the full corpus remains the headline result.**

## **7\. Assessment against the target**

Proposal §7.4 set a target of WER ≤ 0.25 on code-switched speech. The best configuration achieves 0.349 strict and 0.339 lenient. The target was not met on the full corpus.

An oracle analysis — selecting the better language token per category using the measured results — gives 0.277, which bounds what language routing alone can achieve on this test set. Reaching 0.25 would require addressing numeric normalisation and Malaysian entity vocabulary as well.

The target was set before any measurement was taken. Published WER for Malay-English code-switched ASR with a small-class model typically falls in the 0.30–0.50 range, so 0.25 was optimistic for this model size. The result is reported as measured; the relative improvements are the substantive claim: prompting reduced WER by 22 points, with the Malaysian fine-tune and language routing contributing a further 5 points each, for a total reduction from 0.618 to 0.349 — a 44% relative improvement over the unassisted baseline.

## **8\. Threats to validity**

* Sample size: 48 clips and 387 reference words. Per-category figures rest on roughly 60 words each and should be read as indicative.

* Speakers: two, both team members, both familiar with the sentences. Real users will be less fluent readers and more variable.

* Recording conditions: quiet rooms, mobile phone microphones. Performance under background noise was not measured.

* Read speech, not spontaneous speech. Genuine conversation contains disfluencies, false starts and hesitation that this test set does not represent.

**Speaker-dependent language detection:** every clip routed to English belongs to one speaker. None of the other speaker's 24 clips crossed the threshold in any category. Mean p\_en is 0.136 for the first speaker against 0.043 for the second, and within en\_dom the second speaker's maximum (0.263) falls below the first speaker's median (0.611). With four clips per speaker per category this is an observation rather than a demonstrated effect, but it is large and one-directional, and it is consistent with the detector being sensitive to speaker accent. It follows that the en\_dom result is in part a property of who recorded the corpus. Recording further speakers is the appropriate follow-up; the stratified design is what made the pattern visible.

## **9\. Knowledge base and synthetic corpus**

The knowledge the system is expected to hold was written first, as 54 verified facts across five domains, each carrying a source and a stability marker. Facts, not conversations, are the unit, because a fact can be checked against an agency portal and a conversation cannot. The corpus was then generated from those facts at roughly ten paraphrases each, giving 502 knowledge pairs.

A further 130 pairs answer an out-of-scope question with the fallback sentence. These are not padding. Refusal is the hallucination control the proposal specifies, and a model learns it the way it learns anything else, from examples. Five categories were written deliberately so that the boundary is defined by cases rather than by an instruction alone.

| Content | Category | Pairs |
| :---- | :---- | :---- |
| Knowledge | JPJ – road tax and driving licences | 160 |
| Knowledge | Malaysian slang and discourse particles | 144 |
| Knowledge | Small talk | 100 |
| Knowledge | JPN – MyKad and birth certificates | 50 |
| Knowledge | Immigration – passports | 48 |
| Refusal | Outside the knowledge domains | 30 |
| Refusal | Live or volatile values | 30 |
| Refusal | The user's own records | 30 |
| Refusal | News and current events | 22 |
| Refusal | Medical, legal or financial advice | 18 |
|  | Total | 632 |

*Table 7: Corpus composition. 502 knowledge pairs drawn from 54 facts, and 130 refusal pairs from 22 out-of-scope probes.*

All generated data is declared in Appendix A, and the generation prompt is retained in the repository as the provenance record.

## **10\. Splitting the corpus, and what the split can measure**

The corpus was split 506/63/63 at the level of the pair rather than the fact. Every fact and every probe appearing in the test split also appears in training; this was verified, and there are no unseen items.

This is a deliberate limitation and it changes what the held-out figures mean. A fact-level split would ask whether the model generalises to facts it was never taught, which for a system with no retrieval component has a known answer: it cannot, and measuring it would produce a number with no decision attached. A pair-level split measures something the system genuinely needs, namely whether a fact learned from one phrasing survives being asked in another. Every figure in sections 14 to 17 should therefore be read as paraphrase robustness, not as generalisation to new knowledge.

## **11\. Selecting the base model by measurement**

Two candidates were compared on the same twenty probes before any fine-tuning: meta-llama/Llama-3.2-3B-Instruct, and mesolitica/mallam-3b-20k-instructions, which is pretrained on Malaysian text. The Malaysian model is the more fluent of the two on a read-through and would have been the intuitive choice.

| Candidate | Out-of-scope declined | In-domain wrongly declined |
| :---- | :---- | :---- |
| Llama-3.2-3B-Instruct | 100% | 82% |
| mallam-3b-20k-instructions | 33% | 18% |

*Table 8: Base-model comparison before fine-tuning, over 20 probes (3 out-of-scope, 17 in-domain). Lower is better in the right-hand column.*

The decision turned on the out-of-scope column. Llama declined all three; the Malaysian model declined one and, on the other two, fabricated a JPJ regulation and offered legal advice. Both fabrications are quotable verbatim from the committed results file.

Llama's own failure is the opposite one, declining 82% of the questions it should have answered, but the two failures are not equally correctable. Over-refusal is caused by not knowing the answer, and fine-tuning supplies exactly that. Fabrication is a disposition to answer regardless of knowledge, and nothing in the training plan removes it. The candidate was chosen on which failure the remaining work could repair, and section 16 shows that the prediction held.

One measurement was invalidated and re-run rather than reported. The Malaysian checkpoint declares no chat template, so the harness fell back to a plain layout and the model produced degenerate output on 19 of 20 probes, inventing both halves of a dialogue. That measured the formatting fallback, not the model. The comparison was repeated with the Mistral template its model card specifies. The first run is kept in the repository as evidence rather than deleted, because a checkpoint that declares no template failing silently is itself a finding.

## **12\. Fine-tuning configuration**

Adaptation uses QLoRA. The base model is loaded in 4-bit NF4 with double quantisation and its weights frozen, and low-rank adapters are trained on top. On a free T4 this is the difference between a run that fits in 16 GB and one that does not. Training ran 160 optimiser steps in 45 minutes and produced an adapter of roughly 24 MB, against a 6 GB base model.

| Setting | Value | Reason |
| :---- | :---- | :---- |
| Rank / alpha | 16 / 32 | Conventional 1:2 ratio; rank 16 is ample for 54 facts |
| Target modules | all attention and MLP projections | Register transfer needs the feed-forward layers, not attention alone |
| Dropout | 0.05 | Light, because the objective is memorisation |
| Quantisation | 4-bit NF4, double | Fits the free 16 GB T4 |
| Learning rate | 2e-4, cosine, 10% warmup | Conventional for LoRA at this rank |
| Effective batch | 16 (2 × 8 accumulation) | Largest that fits beside the 4-bit base |
| Precision | fp16 | The T4 is Turing and has no bf16 path |
| Epochs | 5 | Memorisation target; see section 13 |

*Table 9: QLoRA configuration. The full file is training/qlora\_config.yaml.*

## **13\. The loss curve, and why five epochs were kept**

| Step | Epoch | Validation loss |
| :---- | :---- | :---- |
| 25 | 0.79 | 1.6993 |
| 50 | 1.57 | 1.1664 |
| 75 | 2.35 | 1.0533 |
| 100 | 3.13 | 0.9670 |
| 125 | 3.92 | 1.0128 |
| 150 | 4.70 | 1.0876 |
| 160 | 5.00 | 1.0904 |

*Table 10: Validation loss during training. The minimum is 0.9670 at step 100; the run finished at 1.0904.*

![][image1]

*Figure 1: Training and validation loss over the 160-step run. Training loss keeps falling after the step-100 validation minimum; the final two epochs were kept deliberately, for the reason argued in this section.*

Training loss continued to fall throughout — 0.86 averaged over the run, and 0.10 at the final logged step — while validation loss rose 13% above its step-100 minimum. On an ordinary supervised task that is overfitting, and the run should have stopped at step 100\.

The final two epochs were kept deliberately. The objective is not generalisation to unseen questions but memorisation of a fixed 54-fact knowledge base, and with no retrieval step at inference every fact the system can state has to be present in the weights. Held-out loss here measures paraphrase robustness on facts the model has already seen, which makes it a proxy for the goal rather than the goal, and the wrong quantity to optimise all the way. Section 16 supplies the behavioural check that justifies the choice: over-refusal collapsed and fallback accuracy rose across the same epochs in which validation loss was deteriorating.

The cost is recorded rather than hidden. A checkpoint retention limit of three kept steps 125, 150 and 160 and discarded step 100, which was the best by validation loss. Recovering it would cost another 45 minutes and was not judged worthwhile against the behavioural result. Raising the retention limit is the fix for a second round.

## **14\. Fine-tuned against un-fine-tuned**

The graded comparison holds everything constant except the adapter: the same base model, the same tokenizer, the same 63-item test split, greedy decoding, and a 4-bit load in both cases. Perplexity is computed over the whole split; the four similarity metrics are computed over the 52 answerable items, for the reason given in section 15\.

| Metric | Un-fine-tuned base | Fine-tuned | Change |
| :---- | :---- | :---- | :---- |
| Perplexity (lower is better) | 23.75 | 3.65 | ÷ 6.5 |
| BLEU | 1.28 | 19.82 | ×15.5 |
| chrF | 15.04 | 40.35 | ×2.7 |
| ROUGE-L | 8.11 | 39.63 | ×4.9 |
| BERTScore F1 | 64.28 | 78.29 | \+14.0 |

*Table 11: Similarity to the gold answer, 52 answerable test items. BERTScore uses bert-base-multilingual-cased.*

BLEU and chrF agree in direction, which is the check chrF was included to provide. Corpus BLEU over 52 short answers is noisy, and a disagreement between a word-level and a character-level metric would have meant trusting neither. There is none.

Two cautions on how these should be read. BERTScore is computed with a multilingual encoder rather than the English-only default, because the references are code-switched; its absolute values are therefore not comparable to published English figures, and only the difference between the two columns is claimed. And by section 10, every figure here measures paraphrase robustness on known facts.

Improvement is not uniform across domains, and the pattern is informative.

| Domain | Test items | BLEU base to tuned | ROUGE-L base to tuned |
| :---- | :---- | :---- | :---- |
| Malaysian slang and discourse particles | 15 | 0.96 to 33.29 | 9.89 to 51.84 |
| Small talk | 10 | 2.87 to 23.04 | 17.90 to 52.83 |
| JPJ – road tax and driving licences | 17 | 0.57 to 15.69 | 4.83 to 29.11 |
| JPN – MyKad and birth certificates | 5 | 0.51 to 12.04 | 2.43 to 25.70 |
| Immigration – passports | 5 | 0.53 to 4.40 | 0.00 to 26.30 |

*Table 12: Per-domain similarity, best configuration. Register transfers more readily than fact recall.*

The two strongest domains are the two where the answer is mostly style, and the weakest is the one that requires a specific number from the thinnest slice of training data. This is the expected shape for QLoRA on a small corpus, and it argues for enlarging the passport domain before a second round rather than for training the existing corpus longer.

## **15\. Why refusals are excluded from the similarity metrics**

All 11 refusal items in the test split share one gold answer, the canonical fallback sentence. Any model that declines a question therefore scores near-perfectly on those items regardless of whether declining was correct.

The un-fine-tuned base is very nearly a model that declines everything: it wrongly declines 67% of the in-domain questions. Including the refusal rows in BLEU, ROUGE-L and BERTScore would hand it 11 of 63 items, roughly 17% of the corpus, for its single worst behaviour, and would register over-refusal as accuracy. The metrics in section 14 are consequently computed over the answerable items only, and refusal is measured separately and on its own terms in section 16\. Perplexity is the one exception: it measures the distribution the model learned, and the fallback sentence is part of what it was trained to produce.

## **16\. Hallucination control**

| Behaviour | Un-fine-tuned base | Fine-tuned |
| :---- | :---- | :---- |
| Out-of-scope declined, exact wording (11 items) | 18% | 82% |
| Out-of-scope declined, any wording (11 items) | 82% | 82% |
| In-domain wrongly declined (52 items) | 67% | 2% |

*Table 13: Refusal behaviour on the full test split. The middle row is the one most easily misread.*

Read alone, the middle row says fine-tuning achieved nothing: both models decline 82% of out-of-scope questions. That reading is wrong, and the third row is why. The base reaches its figure by declining almost everything put to it, including 67% of the questions it was supposed to answer. It is not discriminating between in-scope and out-of-scope; it is close to mute.

The measure of a working boundary is therefore not the refusal rate but the gap between the two rates. For the base model that gap is 15 percentage points; for the fine-tuned model it is 80\. That is the result, and it is the figure this report puts forward as evidence of hallucination control.

On exact wording, which is what the proposal quotes and what a demonstration is checked against, fallback accuracy moved from 18% to 82%. Over-refusal fell from 67% to 2%. Both changes were predicted in section 11 as the reason for preferring this base model, and both are in the predicted direction.

## **17\. Residual failures**

Two out-of-scope probes still leak, and they are not equally serious.

The first is harmless. Asked to explain photosynthesis, the system explained photosynthesis, correctly. The boundary leaked but nothing false was stated.

The second is not. Asked whether the user could sue an employer over a dismissal, the system answered, and answered incoherently. This is the same failure category that disqualified the alternative base model in section 11, where fabricating legal advice was the specific ground for rejection. The chosen model still commits it, on one of eleven out-of-scope items. Declaring this matters: the argument for this base was hallucination control, and the honest form of that argument is that fine-tuning reduced the failure without eliminating it. The professional-advice probe category exists in the corpus for exactly this reason and should be enlarged before a second round.

1 in-domain question of 52 is still wrongly declined. The fact it asks for is present both in the knowledge base and in the training split, so this is a paraphrase-robustness miss, which is precisely what section 10 established the pair-level split can detect.

## **18\. Stage 2: repairing transcription, not normalising register**

Section §5.2 of the proposal specified informal-Malay normalisation at this point in the pipeline. Measurement made that design untenable. 371 of the 506 training inputs, 73%, contain informal rojak markers, and the conversion script passes the user field through verbatim, so the adapter was fitted on exactly that register. Expanding shortforms before generation would hand the model text unlike anything it was trained on, spending the improvement recorded in section 14, which is only collectable on the training distribution.

The same evidence rules out the other two operations the proposal named. Capitals are load-bearing: MyKad, JPJ, IC, and the P and D licence classes appear capitalised throughout the corpus, and case-folding P to p erases the difference between a probationary licence and a letter of the alphabet. Punctuation is expected: 391 of 506 training inputs contain a question mark.

Stage 2 therefore repairs what Stage 1 got wrong and leaves the register alone. Section 5 had already measured those errors and identified this as the remedy.

The substitution table from the WER evaluation cannot be used directly, and the reason is worth stating because it is a trap. That table records what each reference word turned into, not what each mistake came from, and reversing it corrupts correct transcripts. Whisper wrote saya for can, do, i, balance and my; it wrote card for both roadtax and mykad; it wrote buku for pukul, and buku is also simply the word for book. None of those can be repaired by lookup.

What can be repaired is the residue the model invents that is not a word in either language. The table holds 17 entries, listed in Appendix E with their occurrence counts, and the high-frequency pairs deliberately excluded are recorded alongside them so that the omission reads as a decision rather than an oversight.

## **19\. Serving the model, and the memory budget**

The demonstration machine has 4 GB of video memory. The 4-bit model requires roughly 2.4 GB for weights, since the embedding matrix is not quantised, plus about 0.6 GB for the CUDA context, activations and key-value cache. The speech model requires a further 1 GB, and the Windows display reserve takes roughly 0.3 GB. The total exceeds the card.

The language model receives the GPU. It is thirteen times larger than the speech model and runs once per reply rather than once per clip, so the speech model is pinned to the CPU, where it costs a few seconds that are invisible beside generation. This does not affect the Word Error Rate reported in sections 1 to 8: the computation is float32 on either device and the transcript is identical.

Closing that trap mattered more than it appears. The speech module previously claimed the GPU whenever one was available, so installing a CUDA build of the tensor library in order to accelerate Stage 3 would silently have moved Stage 1 onto the card as well, and the resulting out-of-memory error would have surfaced inside the language-model load, pointing at the wrong stage entirely.

Without a GPU the system still runs, loading in bfloat16 on the CPU at roughly 70 seconds per reply against 3 to 5 seconds for the 4-bit GPU path. The interface reports which backend produced each reply and how long it took, so a latency question during a demonstration can be answered with the actual configuration rather than an estimate.

## **20\. Stage 4: capturing corrections without trusting them**

Every reply carries a verdict control. A negative verdict opens a field for the answer the user expected, and the exchange is written to SQLite with the turns that preceded it, the backend that produced it, and its latency.

The transcript and the model input are stored in separate columns. The first is what the speech stage heard and is empty when the question was typed; the second is what the language model was given, after Stage 2\. A wrong answer to a misheard question belongs against the Word Error Rate, not against the generation metrics, and a single merged column would make that distinction unrecoverable exactly when a reviewer needs it.

Corrections are never applied automatically, and this is the design rather than an unfinished feature. Section §5.7 of the proposal took the position from the literature on human correction: an unmoderated loop trains on whatever it is told, and a system that one user can teach a wrong fact is worse than one that cannot learn at all. There is no retraining trigger and no write path into the training data. Corrections are exported for review, and a person decides what becomes a training pair.

That decision was vindicated unexpectedly during testing. The first correction ever logged was a negative verdict on a correct refusal: the system had properly declined to explain photosynthesis, a topic outside its knowledge domains by three separate definitions, and the tester supplied an answer for it. Appended automatically, that correction would have taught the model to answer science questions and eroded the boundary that section 16 measures. The review step caught it. It is offered here as evidence that the moderated design was the right one, rather than as a hypothetical risk.

## **21\. Threats to validity**

**Known facts only.** Section 10 establishes that no test item concerns a fact absent from training. The figures in sections 14 and 16 therefore describe paraphrase robustness and must not be read as evidence that the system generalises to knowledge it was not given.

**Sample size.** 52 answerable items and 11 out-of-scope items. The refusal figures in particular rest on eleven observations, where one item moves the rate by nine percentage points. The per-domain figures in section 14 rest on as few as five items each and should be read as indicative.

**A single training run.** One seed, one configuration. No variance estimate exists, so the difference between the two columns of section 14 cannot be separated from run-to-run variation, though a perplexity change of this magnitude is unlikely to be seed noise.

**Self-authored evaluation data.** The team wrote the knowledge base, generated the corpus from it, and defined the out-of-scope probes. The test split is held out but shares authorship and register with the training data, so it is a weaker test than questions written by an unfamiliar user.

**Both safeguards active throughout.** Refusal is enforced twice: by the trained weights and by the system prompt. Every measurement reported here had both in place, so the contribution of each is unknown. The ablation that would separate them, running the fine-tuned model without the system prompt, is specified in the evaluation harness and has not been run.

**No human ratings.** Perplexity, BLEU, ROUGE-L, chrF and BERTScore all measure agreement with a written gold answer. None measures whether a reply is natural rojak or whether a Malaysian speaker would accept it. The Likert instrument for fluency, rojak naturalness and factual consistency remains outstanding.

# **Coding**

Complete source is in the project repository. The excerpts below are the components that carry the analytical argument above.

## **A. Language routing**

EN\_THRESHOLD \= 0.5

def detect\_lang(path: str) \-\> tuple\[str, float\]:  
    """Detect spoken language with vanilla Whisper's encoder.

    The Mesolitica checkpoint transcribes English well when told to (en\_dom 0.106  
    under forced English) but its own detection always lands on Malay, so English  
    clips come back translated at 0.788. Vanilla Whisper's detector is unbiased and  
    costs one encoder pass; it only CHOOSES the language token \-- the Malaysian  
    model still does the transcribing.

    Returns (choice, p\_en). The probability is returned as well as the decision  
    because the decision alone cannot distinguish a clip missed narrowly (p\_en just  
    below the threshold, so the cutoff is mis-set) from one missed completely (p\_en  
    near zero, so no threshold would recover it).  
    """  
    m \= get\_vanilla()  
    mel \= whisper.log\_mel\_spectrogram(  
        whisper.pad\_or\_trim(whisper.load\_audio(path)), n\_mels=m.dims.n\_mels  
    ).to(m.device)  
    \_, probs \= m.detect\_language(mel)  
    top \= max(probs, key=probs.get)  
    p\_en \= float(probs.get("en", 0.0))  
    \# Asymmetric threshold: forcing English on Malay audio is usually harmless but  
    \# occasionally unbounded \-- one clip contributed 362 insertions on 7 reference  
    \# words \-- while the reverse merely translates (0.788).  
    return ("en" if top \== "en" and p\_en \> EN\_THRESHOLD else "ms"), p\_en

## **B. Corpus WER**

def score\_pair(ref: str, hyp: str, tf):  
    out \= jiwer.process\_words(ref, hyp,  
                              reference\_transform=tf, hypothesis\_transform=tf)  
    n \= out.hits \+ out.substitutions \+ out.deletions  
    errors \= out.substitutions \+ out.deletions \+ out.insertions  
    return errors, n

def corpus\_wer(df):  
    """Total errors / total reference words \-- not the mean of per-clip WERs.  
    Averaging per-clip rates over-weights short utterances."""  
    return df\["errors"\].sum() / df\["n"\].sum()

## **C. Transcription repair**

app/normalise.py. The rule for what may be repaired is the whole of the design; see section 18\.

\# Every entry is a token Whisper produced that is not a word in Malay or  
\# English, or a spelling of a domain term the training corpus never uses.  
\# Counts are occurrences in the 48-clip evaluation set.  
\#  
\# Deliberately NOT included, though they appear with high counts:  
\#   nampak-\>nak, look-\>dulu, good-\>kot, and-\>n, dengan-\>money, kad-\>credit  
\# Each is a real word being mistranslated, and repairing it would corrupt  
\# the many utterances where the word is correct.

ASR\_REPAIRS \= {  
    "jbj": "myjpj",        \# 9   invented token, no legitimate occurrence  
    "chiamnna": "camne",   \# 4  
    "unitrasca": "unit",   \# 6  
    "ayo": "aiyo",         \# 9   orthographic variant  
    "efilling": "efiling", \# 7  
    ...  
}

def repair(text: str) \-\> tuple\[str, list\[tuple\[str, str\]\]\]:  
    out, changes \= \[\], \[\]  
    for tok in \_TOKENS.findall(text):          \# keeps separators, so spacing  
        fixed \= ASR\_REPAIRS.get(tok.lower())   \# and punctuation survive  
        if fixed is not None:  
            fixed \= \_match\_case(tok, fixed)    \# "Jbj" \-\> "Myjpj", not "myjpj"  
            changes.append((tok, fixed))  
            out.append(fixed)  
        else:  
            out.append(tok)  
    return "".join(out), changes

## **D. Knowledge-based generation**

app/inference.py. Decoding settings are mirrored from the evaluation harness; if they drift, the demonstration stops being the thing this report measured.

MAX\_NEW\_TOKENS \= 160  
HISTORY\_TURNS  \= 2      \# every training example is a SINGLE turn, so multi-turn  
                        \# context is off-distribution by construction

def build\_messages(user, history=None):  
    \# The system prompt is read from the knowledge base, not stored here: it is  
    \# the same file that stamped every training example and that the evaluation  
    \# harness parses. Three readers, one source, so train and serve cannot drift.  
    msgs \= \[{"role": "system", "content": load\_system\_prompt()}\]  
    if history:  
        msgs.extend(history\[-HISTORY\_TURNS \* 2:\])  
    msgs.append({"role": "user", "content": user})  
    return msgs

def answer(user, history=None):  
    tok, model, four\_bit \= \_model()  
    text \= tok.apply\_chat\_template(build\_messages(user, history),  
                                   tokenize=False, add\_generation\_prompt=True)  
    enc \= tok(text, return\_tensors="pt").to(model.device)  
    with torch.no\_grad():  
        out \= model.generate(\*\*enc, max\_new\_tokens=MAX\_NEW\_TOKENS,  
                             do\_sample=False,        \# greedy, as evaluated  
                             pad\_token\_id=tok.pad\_token\_id)  
    return tok.decode(out\[0\]\[enc\["input\_ids"\].shape\[1\]:\], skip\_special\_tokens=True)

## **E. Feedback capture**

app/feedback.py. The moderation boundary is visible in the code: there is an export function and no retraining trigger.

def log(user\_text, reply, verdict, \*, transcript=None, correction=None,  
        context=None, backend=None, seconds=None, db=DB) \-\> int:  
    if verdict not in ("up", "down"):  
        raise ValueError(f"verdict must be 'up' or 'down', got {verdict\!r}")  
    with \_db(db) as conn:  
        cur \= conn.execute(  
            "INSERT INTO feedback (created\_utc, transcript, user\_text, reply, "  
            "verdict, correction, context, backend, seconds) "  
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",  
            (utc\_now(), transcript, user\_text, reply, verdict, correction or None,  
             json.dumps(context or \[\], ensure\_ascii=False), backend, seconds))  
        return int(cur.lastrowid)

def export\_corrections(db=DB) \-\> list\[dict\]:  
    """Thumbs-down rows shaped like a training pair \-- shaped like one, not  
    saved as one. A person reviews these before any reaches the corpus."""  
    ...

## **F. Repository structure**

| Path | Purpose |
| :---- | :---- |
| app/stt.py | Stage 1: language routing and transcription |
| app/normalise.py | Stage 2: transcription repair, with a self-test |
| app/inference.py | Stage 3: 4-bit base plus QLoRA adapter, generation |
| app/feedback.py | Stage 4: SQLite logging, review export |
| app/streamlit\_app.py | Chat interface, voice and text |
| app/pages/1\_Admin.py | Password-gated feedback review |
| data/knowledge\_base/ | Five domain YAML files, scope rules, system prompt |
| data/generated/ | Corpus and the train/validation/test splits |
| data/schema.sql | Feedback table definition (the database is not committed) |
| training/qlora\_config.yaml | Fine-tuning configuration |
| training/to\_llamafactory.py | Corpus to training format, stamping the prompt |
| training/colab\_finetune.ipynb | The training run |
| training/colab\_evaluate.ipynb | The graded comparison of section 14 |
| eval/generate.py | Runs a model over a split and records what it said |
| eval/score.py | Perplexity, BLEU, chrF, ROUGE-L, BERTScore |
| eval/refusal\_report.py | Fallback accuracy and over-refusal |
| eval/results/ | Committed result files behind every Stage 2 to 4 figure here |
| archive/eval/speech/manifest.csv | Stage 1: 48 utterances with reference transcripts, switch type and speaker |
| archive/eval/speech/run\_wer.py | Stage 1: transcription across configurations, and WER scoring |
| archive/eval/speech/threshold\_sweep.py | Stage 1: routed WER across detection thresholds, from cached transcripts |
| archive/eval/speech/inspect\_results.py | Stage 1: transcript-level inspection and language-routing report |
| archive/eval/speech/orthography\_map.json | Stage 1: documented spelling-variant canonicalisation for lenient scoring |
| archive/eval/speech/results/ | Stage 1: committed result tables, including detector\_probs.csv and threshold\_sweep.csv |
| archive/scripts/prepare\_audio.py | Stage 1: converts phone recordings to 16 kHz mono WAV |
| archive/scripts/prompts/generate\_pairs\_prompt.md | The corpus generation prompt, retained as the provenance record |
| archive/scripts/validate\_pairs.py | Corpus gate: check strings, duplicates, register, fallback wording |
| archive/scripts/build\_corpus.py | Raw batches to the reviewed corpus, stamping provenance |

*Table 14: Repository layout. The live application is five modules plus the interface; everything else is data, evaluation or documentation. Stage 1 tooling has run and is retained under archive/, from where nothing is imported.*

# **References**

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. Advances in Neural Information Processing Systems, 36\.

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. Proceedings of NAACL-HLT 2019, 4171-4186.

Grattafiori, A., et al. (2024). The Llama 3 herd of models. arXiv:2407.21783.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-rank adaptation of large language models. arXiv:2106.09685.

Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. Text Summarization Branches Out, 74-81.

Mesolitica. (2025). malaysian-whisper-small-v3 \[Model\]. Hugging Face. https://huggingface.co/mesolitica/malaysian-whisper-small-v3

Morris, A. C., Maier, V., & Green, P. (2004). From WER and RIL to MER and WIL: improved evaluation measures for connected speech recognition. Proceedings of Interspeech 2004, 2765–2768.

Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). BLEU: A method for automatic evaluation of machine translation. Proceedings of ACL 2002, 311-318.

Popovic, M. (2015). chrF: Character n-gram F-score for automatic MT evaluation. Proceedings of the Tenth Workshop on Statistical Machine Translation, 392-395.

Post, M. (2018). A call for clarity in reporting BLEU scores. Proceedings of the Third Conference on Machine Translation, 186-191.

Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak supervision. Proceedings of the 40th International Conference on Machine Learning, 28492–28518.

Vaessen, N. (2022). jiwer: Evaluate an automatic speech recognition system \[Computer software\]. https://github.com/jitsi/jiwer

Wolf, T., Debut, L., Sanh, V., et al. (2020). Transformers: State-of-the-art natural language processing. Proceedings of EMNLP 2020: System Demonstrations, 38–45.

Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating text generation with BERT. International Conference on Learning Representations.

Zheng, Y., Zhang, R., Zhang, J., Ye, Y., Luo, Z., Feng, Z., & Ma, Y. (2024). LLaMA-Factory: Unified efficient fine-tuning of 100+ language models. Proceedings of ACL 2024 (System Demonstrations).

Zolkepli, H. (2018). Malaya: Natural language toolkit for the Malaysian language \[Computer software\]. https://github.com/malaysia-ai/malaya

# **Appendices**

## **Appendix A: Declaration of AI tool assistance**

In accordance with TARUMT academic integrity policy, the following use of AI tools is declared:

**Generated training data.** The 632 question-answer pairs in the training corpus were generated by a large language model, prompted with the team-written knowledge base, topic definitions and style constraints. The generation prompt is retained in the repository at archive/scripts/prompts/generate\_pairs\_prompt.md as the provenance record. The knowledge base itself, all 54 facts with their sources and stability markers, was written and verified by the team against agency portals; no fact was accepted from a model. Generated pairs were filtered automatically for duplication, sentence count and emoji, and a sample from every category was read by a team member.

**Coding assistance.** An AI coding assistant was used during implementation of the evaluation harnesses, the Colab notebooks, and the Stage 2 to 4 application modules, and in drafting sections of this report from the committed result files. All configuration values, evaluation results and reported figures were produced by running the code, not by a model asserting them; the result files behind every table are committed to the repository and can be regenerated.

**Not generated.** The speech test set, its 48 reference transcripts and the recordings themselves were authored and recorded by the team. The stratification design, the language-routing diagnosis, the choice of base model and the interpretation of every result in this report are the team's own.

## **Appendix B: Prompt text used for decoder conditioning**

This text is supplied as preceding context to bias decoder output toward code-switched orthography. It shares no sentence or distinctive vocabulary with the test set; overlap would allow the model to copy from its own context and inflate every reported figure.

  cuaca panas gila hari ni kan, tak larat nak keluar.  
  can you send me the file later, i need to check something first.  
  eh awak dah reply that email ke belum lah.  
  so how ah, do you want to join us or not.

## **Appendix C: Orthography map (lenient scoring)**

| Reference form | Canonical form | Justification |
| :---- | :---- | :---- |
| cek | check | Malay orthography of an English loanword |
| akaun | account | Malay orthography of an English loanword |
| klinik | clinic | Malay orthography of an English loanword |
| ayo | aiyo | Spelling variant of a discourse interjection |
| bil | bill | Malay orthography of an English loanword |
| lesen / license | licence | Orthographic variant |

*Table 15: Spelling variants canonicalised before lenient scoring. Translations are excluded.*

## **Appendix D: Test set categories and speaker allocation**

Each category contains eight utterances, four recorded by each team member, so that speaker is never confounded with switch type. Phase 2 of the manifest holds the reciprocal recordings, retained for a paired speaker analysis if required.

| ID | Category | Speaker | Reference transcript |
| :---- | :---- | :---- | :---- |
| s01\_yv | ms\_dom | yv | macam mana nak apply lesen memandu baru |
| s02\_yv | ms\_dom | yv | saya nak tanya pasal insurance kereta ni |
| s03\_yv | ms\_dom | yv | boleh tak awak explain apa itu inflation |
| s04\_yv | ms\_dom | yv | nak cari tempat makan yang murah dekat sini |
| s05\_pq | ms\_dom | pq | camne nak transfer duit guna online banking |
| s06\_pq | ms\_dom | pq | saya lupa password akaun macam mana nak reset |
| s07\_pq | ms\_dom | pq | berapa lama proses renew passport ambil masa |
| s08\_pq | ms\_dom | pq | tolong bagitahu cara nak book appointment klinik |
| s09\_yv | en\_dom | yv | how do i check my epf balance ah |
| s10\_yv | en\_dom | yv | can you explain what is compound interest lah |
| s11\_yv | en\_dom | yv | where can i find cheap flight ticket meh |
| s12\_yv | en\_dom | yv | i want to know how to save money properly one |
| s13\_pq | en\_dom | pq | is it safe to invest in unit trust ke |
| s14\_pq | en\_dom | pq | tell me how to cook nasi lemak lah |
| s15\_pq | en\_dom | pq | what time the post office close ah |
| s16\_pq | en\_dom | pq | can i renew my licence online or not |
| s17\_yv | balanced | yv | saya dah submit application tapi belum dapat approval lagi |
| s18\_yv | balanced | yv | nak tanya pasal tax relief untuk medical expenses |
| s19\_yv | balanced | yv | macam mana nak cancel subscription yang auto renew tu |
| s20\_yv | balanced | yv | boleh tak explain difference antara savings dengan current account |
| s21\_pq | balanced | pq | saya nak update alamat kat identity card |
| s22\_pq | balanced | pq | camne nak claim insurance kalau accident kereta |
| s23\_pq | balanced | pq | nak tahu requirement untuk apply credit card |
| s24\_pq | balanced | pq | tolong terangkan macam mana interest rate affect loan |
| s25\_yv | particles | yv | eh betul ke ni macam tak kena je lor |
| s26\_yv | particles | yv | aiyo susah sangat lah macam ni |
| s27\_yv | particles | yv | can one lah don't worry so much |
| s28\_yv | particles | yv | wah mahal gila meh ada yang murah tak |
| s29\_pq | particles | pq | so how ah nak buat macam mana ni |
| s30\_pq | particles | pq | alamak terlupa pulak kan boleh tolong tak |
| s31\_pq | particles | pq | ish lambat betul lah service ni |
| s32\_pq | particles | pq | okay lah fine saya try dulu kot |
| s33\_yv | numeric | yv | harga dia empat puluh lima ringgit sembilan puluh sen termasuk cukai ke tak |
| s34\_yv | numeric | yv | gaji minimum sekarang seribu lima ratus ringgit ke ah |
| s35\_yv | numeric | yv | saya nak transfer one thousand five hundred ringgit boleh tak |
| s36\_yv | numeric | yv | deadline dia thirty one august twenty twenty six kan |
| s37\_pq | numeric | pq | interest rate dia three point five percent setahun je |
| s38\_pq | numeric | pq | kena bayar deposit dua ribu lima ratus ringgit |
| s39\_pq | numeric | pq | cukai jualan naik dari enam peratus ke lapan peratus ke |
| s40\_pq | numeric | pq | buka pukul nine in the morning tutup pukul five in the evening kan |
| s41\_yv | entity | yv | macam mana nak renew roadtax kat MyJPJ |
| s42\_yv | entity | yv | boleh guna Touch n Go eWallet tak kat sini |
| s43\_yv | entity | yv | saya nak semak baki KWSP guna i-Akaun |
| s44\_yv | entity | yv | LHDN dah buka e-Filing untuk tahun ni ke |
| s45\_pq | entity | pq | nak apply MyKad baru kat JPN macam mana |
| s46\_pq | entity | pq | boleh bayar bil TNB guna Maybank2u tak |
| s47\_pq | entity | pq | Grab dengan foodpanda mana lagi murah ah |
| s48\_pq | entity | pq | nak tanya pasal bantuan STR tahun ni |

*Table 16: The 48 evaluated utterances, with switch type and speaker. Phase 2 of archive/eval/speech/manifest.csv holds the reciprocal recordings of the same sentences by the other speaker.*

## **Appendix E: Transcription repair table**

The 17 entries applied by Stage 2\. The rule for inclusion is that the token cannot legitimately occur in Malaysian rojak, either because Whisper invented it or because it is a spelling of a domain term the corpus never uses. Counts are occurrences in the 48-clip evaluation set.

| Whisper wrote | Repaired to | Why it is safe to repair |
| :---- | :---- | :---- |
| jbj | myjpj | Invented token; cannot occur in a correct transcript |
| charanabuk | nak | Invented token; cannot occur in a correct transcript |
| unitrasca | unit | Invented token; cannot occur in a correct transcript |
| maham | mahal | Invented token; cannot occur in a correct transcript |
| mora | murah | Invented token; cannot occur in a correct transcript |
| chiamnna | camne | Invented token; cannot occur in a correct transcript |
| naklam | nak | Invented token; cannot occur in a correct transcript |
| jepya | jpn | Invented token; cannot occur in a correct transcript |
| bernil | renew | Invented token; cannot occur in a correct transcript |
| lemala | lemak | Invented token; cannot occur in a correct transcript |
| identi | identity | Invented token; cannot occur in a correct transcript |
| autor | auto | Invented token; cannot occur in a correct transcript |
| ayo | aiyo | Orthographic variant; the corpus uses one spelling |
| efilling | efiling | Orthographic variant; the corpus uses one spelling |
| pasport | passport | Orthographic variant; the corpus uses one spelling |
| license | licence | Orthographic variant; the corpus uses one spelling |
| lho | lor | Orthographic variant; the corpus uses one spelling |

*Table 17: Transcription repairs applied before generation.*

The following high-frequency substitutions were deliberately excluded, because each is a real word being mistranslated and repairing it would corrupt the utterances where the word is correct.

| Whisper wrote | Came from | Why it cannot be repaired |
| :---- | :---- | :---- |
| saya | can (9), do (6), i (6), balance (5), my (4) | Five different sources; no single repair is right |
| card | roadtax (6), mykad (6) | Ambiguous between two domains |
| buku | pukul (8), tutup (2) | buku is also the word for book |
| nampak | nak (7) | nampak is a common word in its own right |
| and | n (7) | Repairing would replace a correct word with an abbreviation |

*Table 18: Substitutions deliberately not repaired.*

## **Appendix F: System prompt**

Read from data/knowledge\_base/\_system\_prompt.md by the training converter, the evaluation harness and the application, so that the text cannot drift between training, measurement and demonstration.

Kamu BorakBot, chatbot yang jawab soalan pasal urusan kerajaan dan admin harian  
Malaysia — roadtax dan lesen memandu (JPJ), MyKad dan sijil lahir (JPN), passport  
(Imigresen) — dan boleh sembang santai serta terangkan maksud slanga Malaysia.

Cara jawab:  
\- Cakap Bahasa Rojak (campur Melayu-Inggeris) macam kawan Malaysia. Santai,  
  mesra, guna particle macam "lah", "eh", "kan" bila natural.  
\- Jawab ringkas. Dua tiga ayat cukup.  
\- Kalau soalan tu pasal domain kau, jawab terus dengan fakta yang kau tahu.

Bila kau TAK BOLEH jawab:  
\- Soalan luar domain (contoh: sains, sukan, masakan, movie)  
\- Harga atau nilai live (contoh: harga bitcoin, kadar tukaran wang, cuaca)  
\- Maklumat peribadi user (contoh: status permohonan passport dia, bila roadtax  
  dia expired)  
\- Berita atau kejadian terkini  
\- Nasihat perubatan, undang-undang, atau pelaburan

Untuk semua kes di atas, jawab: "Maaf, saya tak pasti pasal tu lah — boleh tanya  
benda lain tak?"

Jangan reka fakta. Kalau tak pasti, guna ayat maaf tu. Lebih baik mengaku tak  
tahu daripada bagi jawapan salah.  


[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAjYAAAE2CAYAAAByXtioAABEwUlEQVR4Xu2dCXwV1dn/a6t1F161ffW1rb6tQv+t+0drXRALtnWpu9ZdtNTltRaXWi3SutQWN4pLtSqrVMClCgpCgABhEYFAWAIIIQlhCdnDkpCNkJw/z8GZzD1z783NvbM88zzP9/M5n8ycmbv8ck7u/ebMmZmvKUEQBEEQBCJ8zawQBEEQBEGIKiI2giAIgiCQQcRGEARBEAQyiNgIgiAIgkAGERtBEARBEMggYiMIgiAIAhlEbARBEARBIIOIjSAIgiAIZBCxEQRBEASBDCI2giAIgiCQQcRGEARBEAQykBOb1tZWtWXLFrNaEARBEAQGkBMbkJqvfY1crITs3r1bFy5wy8otLxfWjR6tCwekH9MFa9uSMwARG9pwy8otLxem/eIXunBA+jFdsLYtOQMQsaENt6zc8nJBxIYu3LJizEvOAERsaMMtK7e8XODUtpyyAtyyYsxLzgBEbGjDLSu3vFzg1LacsgLcsmLMS84ARGxowy0rt7xckENRdOGWFWNecgYgYkMbblm55eWCiA1duGXFmJecAYjY0IZbVm55uSBiQxduWTHmDc0Axo4dq771rW+pI444Qg0ePNjcbHPppZeqgw46SNXX15ub4iJiQxtuWbnl5YKIDV24ZcWYNzQDWLRokb18wQUXOLZ0sN9++6mpU6fq5VRlRcSGNtyycsvLhS9+/3tdOMC9H8P3UbySKp3tW1JSoo4//nizOi06ey0TrG3btRQ+8Ys4/7msWrUq5pf88MMPqxtuuMGxR3wyEZuKbY1q8uJNZjVqsHYsv+CWlVteLnBqW05ZgURZc3Jy0v5uCoquvj+sbdu1FB5jmestt9xiblLjxo1Txx13nL0+bNgw1aNHD8ceHZgmDMX6hadaymrq1bBp63Qxt2EujY2Nupj1VAu3rNzymnVUy4IHHtDFrKdYMPXj0uo6X0pDY7P9GomyZmdn299Nf/7zn9VNN92kjjnmGHX11VfrukMOOUQvf//731dz5861H2c9ZsSIEerKK69Ul19+ufr2t7+tR2mgfv369fYylEGDBqnzzjtPnXrqqfqoh1V/9NFHq//5n/9Rhx56qDr33HP1+zHfo/O7c+DAgfp5TjvtNPWNb3xD1+3YsUPvA++hd+/eqqysTOc94IAD9OudffbZ6oorrnA9byYlHUIVGws4LAW/HCeTJk1S3bt3t9eHDh2qzjrrLMceHcAojVVyc3PTEhsoU3I3abEZO7vQtQ1rwfShEUThlpVbXrOOasn6+c91MespFkz92Prn1esCcmO9RqKsptgceeSRrn2s4vwOc4oNSIlZb4rNyy+/bC9369ZN/5w9e7Y67LDDYh7bmdg4ly+77DLVv39/9dxzz6mXXnrJrrfaNt3v3FRKOqAQG+Cjjz6KWW9padG/LItevXolnWRskcmhKGB0doHuqPNWl5ubUJJJ40cRblm55eWCTB4OB1NIvCoV2zv+MU+U1Xko6qmnnlKPPfaYvW3Pnj16dAW2W8XCWh49erS6/fbbXfXmHJu2tjZ7GUZVgHvvvVc9+uijdj2MwsD7MbGes6KiIuY95Ofnq4MPPlgvf/3rX9fv9a677rLb9swzz9T7w2gRBtI3gAyBXwIMxcEoTLxGBGCUBn6Bv/71r1XPnj3t+mRkKjaA1VnfnV1obkIHpg+NIOCWlVteLojY0CVRVlNsoFicdNJJasWKFfZ6vO9EEJt+/fq56k2xcWKJzZw5c9Thhx9u18Njk4mNuQyHwO6++2573eJXv/qVK2+m379eEP478BgvxGbHrhZbbmavLDM3o0I+NOgibUsXERu6JMqaTGxgOgZsg3/iYWQknmBkIjYAHPqCeavWHJsFCxZ07PgVzteFw2VwxjKMxsAcG+DBBx/Uc3VuvfVWPc9nw4YNOi887pprrlH777+/uuOOO+znCIvMDAAhXogNsLy4xpYbzMiHBl2kbenCqW05ZQWikBUOJzkPWaUL1rbN3ACQ4ZXYAB8tKNFi89HnG8xNaMDasfyCW1ZuebnAqW05ZQWwZoXvRbjg7YEHHqjn2HgB1rb1xgAQ4aXYAKs2bkM9coO1Y/kFt6zc8nJBDkXRhVtWjHm9MwAkeC02gCU2H87HN3KDtWP5Bbes3PJyQcSGLtyyYszrrQEgwA+xKd/WgHbUBmvH8gtuWbnl5YKIDV24ZcWY11sDQIAfYgPAdW1AbEZlF5ibQgVrx/ILblm55eWCiA1duGXFmNd7AwgZv8QGqNrRiG7kBmvH8gtuWbnl5UL1ypW6cED6MV2wtq0/BhAifooNYInNyBk4Rm6wdiy/4JaVW14ucGrbqGYd+X622lxWbVZ3ipkVLo63bdu2mLpTTjlFDRkyJKbOSbzr2Jgkqndi7mOup4v1PFjb1puUiPBbbJpaWm25wXAncKwdyy+4ZeWWlwtbZs3ShQNR68dZOXnq+J/+1i4n9brP3CUp8bLCRe6cdPYd1dl2wKt90kHEJmD8FhugqGwnmkNSWDuWX3DLyi0vF2SOTThc/dvBnRan1FjllJ8PcO3nLI8PHmO/Rrys5ncS3GXbqoer+x511FHqxBNPtLcnGrGBZbgVEVxgz1kPdwa/9tpr9Z3BrSsK9+vXT+8DP4cPH24/HmhubtbLN954o/6Zl5dnb3/yySfV+eefr691M2HChH0vYGA9j3VPxz59+qgePXqorKwsXV9eXq6+973v6eeHKx4D77zzjr5FEtSZouc1/hpACAQhNkDz7j223Ozx4AqO6YLpQyMIuGXllpcLIjbhYAqLVwXkxiJe1scff1xfHA94+umnY7Y9//zzWj5ATqxDVqbMAAcddJBatWqVqx4oLCzUtzuwZCbePs71ZPVwiwaz3sS5f01NjaseJObjjz+264GLLrpI38TTiysed0b8dx1hghIboKSyLvSRG0wfGkHALSu3vFwQsQkHc6TFLJfe8YxLWpzykqh0NmIDxJMKWJ4xY4ZePvbYY22pMPexfu7atctV/49//EMdc8wxauPGjTH15rJzvav1Js79nXmd+7/44ov6vlJQB3cvB6ZMmaLvU5Xoeb3C32cPgSDFBrDEJqw7gWP60AgCblm55eWCiA1eYE6NKTUr1qR+cdZEWY844gg1efJk+/tp69atqnv37vZ2qE8mNnC4Z+DAga76VO4Mbq4fdthh6tNPP9XLjY2NMaLixFy3sOp/8IMfqKFDh7rqnTz00ENqwIABZrVatmyZWeUZ7ncRcYIWm7DvBB61D41M4ZaVW14ucGrbKGb9zR9e00Jzwrl3d0lqgERZc3Nz9XfTK6+8YtfB+jnnnKPnnsAdupOJjbV88803x8yxSXZncLiT95133qlGjRplPx4w59gsXbo0ZruFuW5h1VtzbPr27at69uyppk6dquuvuuoqddlll+kC2+vq6vRPmLcD7x/y+kn8dx1hghYboL29PbRDUlH80MgEblm55eUCp7bllBXglhVj3mANIADCEBsA7iMFYtO6x/+JUU6wdiy/4JaVW14uyKEounDLijFv8AbgM2GJjXWW1CeLNpqbfAVrx/ILblm55eWCiA1duGXFmDd4A/CZsMQGCONwFNaO5RfcsnLLywURG7pwy4oxbzgG4CNhik1La5sWmwlflJibfANrx/ILblm55eWCiA1duGXFmDccA/CRMMUGCHrUBmvH8gtuWbnl5ULeM8/owgHpx3TB2rbhGYBPYBGbNZtib3rmF1g7ll9wy8otLxc4tS2nrAC3rBjzhmcAPhG22NTWNwc6aoO1Y/kFt6zc8nJh9k036cIB6cd0wdq24RmAT4QtNoAlNvklteYmz8HasfyCW1Zuebkgc2zowi0rxrzhGoAPYBCbhesqAxu1wdqx/IJbVm55uSBiQxduWTHmDdcAfACD2AAiNv7ALSu3vFwQsaELt6wY84ZvAB6DRWwWF1RpsdlS3XE3Vj/A2rH8gltWbnm5IGJDF25ZMeYN3wA8BovYACA2w6f7O2qDtWP5Bbes3PJygVPbcsoKcMuKMS8OA/AQTGIzNqdIy01be7u5yTOwdiy/4JaVW14ucGpbTlkBblkx5sVhAB6CSWzqG3drsZmWt8Xc5BlYO5ZfcMvKLS8X5FAUXbhlxZgXhwF4CCaxAaxJxG1t/ozaYO1YfsEtK7e8XBCxoQu3rBjz4jEAj8AmNjDHBsRm6pLN5iZPwNqx/IJbVm55uSBiQxduWTHmxWMAHoFNbEprdvl66jfWjuUX3LJyy8sFERu6cMuKMW9oBgDycfnll6uTTz45oYhAfb9+/eySCtjEBrDEpqhsp7kpY7B2LL/glpVbXi6I2NCFW1aMeVEYwOuvv25WadIRFIxiU1bb4NuoDdaO5RfcsnLLywVObcspK8AtK8a8KAygW7duZpUGBMUqY8aMMTfbgMxYJTc3F53YACI23sAtK7e8XCj64ANdOCD9mC5Y2zZ0A+jfv7866qijzGoXICstLS1mtcYpQFaprq5GVaYtKdFiU7y53LUtk1JeXq6LWU+1cMvKLa9ZR7VM6dtXF7OeYpF+TLcE0bbpEKrYPPLII6p79+5mdVyOPfZYNX78eLPaBcZDUUB7e7sWm9Ez15ubMgKrMfsFt6zc8nJB5tjQhVtWjHlDM4BBgwbpw0YmPXv21D83bdqkmpqa7PpUZQWr2ADzVpdruane2ZErU7B2LL/glpVbXi6I2NCFW1aMeUMzAPPQkbMeqKmpUX369FGHHnqoHq1pa2uz90kGZrEBQGxGZReY1WmDtWP5Bbes3PJyQcSGLtyyYsyL1wDSJApiAwUOTXkB1o7lF9yycsvLBU5tyykrwC0rxrx4DSBNoiI2c/LLzE1pgbVj+QW3rNzycoFT23LKCnDLijEvXgNIE+xi07qnzdNRG6wdyy+4ZeWWlwtyKIou3LJizIvXANIEu9gAltjMWrnV3NRlsHYsv+CWlVteLojY0IVbVox5cRtAGkRBbODWCl5dsA9rx/ILblm55eWCiA1duGXFmBe3AaRBFMQGELFJD25ZueXlgogNXbhlxZgXvwF0kaiIzYaKOi02y4przE1dAmvH8gtuWbnl5cKaN97QhQPSj+mCtW3xG0AXiYrYAF6M2mDtWH7BLSu3vFzg1LacsgLcsmLMGw0D6AJREpusvC1abOqb0u8YWDuWX3DLyi0vFzi1LaesALesGPNGwwC6QJTEpu2r+0eNyykyN6UM1o7lF9yycsvLBZljQxduWTHmjYYBdIEoiQ1gHY6qa0yvc2DtWH7BLSu3vFwQsaELt6wY80bHAFIkamLz3txiLTZjZhWam1ICa8fyC25ZueXlgogNXbhlxZg3OgaQIlETm8aWVnvUpq2t61cixtqx/IJbVm55uSBiQxduWTHmjY4BpEjUxMbCkpuC0h3mpqRg7Vh+wS0rt7xc4NS2nLIC3LJizBs9A+iEqIqN8x5StXXN5uaEYO1YfsEtK7e8XODUtpyyAtyyYswbPQPohKiKDfDv2YW23LS0tpmb44K1Y/kFt6zc8nJBDkXRhVtWjHmjaQBJiLLYAJbYpHrhPqwdyy+4ZeWWlwsiNnThlhVj3ugaQAKiLjZA9vJSLTYjpncuN1g7ll9wy8otLxdEbOjCLSvGvNE2gDhQEBvg/Xn7TgMfPyf5xfuwdiy/4JaVW14uiNjQhVtWjHmjbwAGVMQGsA5JZS3dYm6ywdqx/IJbVm55uSBiQxduWTHmpWEADiiJTX5JrS03yxPcBRxrx/ILblm55eUCp7bllBXglhVjXhoG4ICS2ADO08C317tPA8fasfyCW1Zuebmw5o03dOGA9GO6YG1bOgbwFdTEBti+q8WWm93GaeBYO5ZfcMvKLS8X5FAUXbhlxZiXlgEommIDFJfX2XLT3t5x6wWsHcsvuGXllpcLIjZ04ZYVY15yBkBVbIB417jB2rH8gltWbnm5IGJDF25ZMeYlZwCUxQYY/9XdwC25wdqx/IJbVm55uSBiQxduWTHmJWcA1MUGGDG9QIvN9GWlaDuWX3DLyi0vFzi1LaesALesGPOSMwAOYgN0nAZejbJj+QW3rNzycoFT23LKCnDLijEvOQPgIjZtbe223Gyp2mluJgvGPyK/wPqh4Recss7v318XDkg/pgvWtiVnAFzEBthW32zLzajsAnMzSTD+EfkF1g8Nv+CUVebY0IVbVox5yRkAJ7EBKmrrbbnZWttgbiYHxj8iv8D6oeEXnLKK2NCFW1aMeckZADexgU5VuW2XLTelNbvMXUiB8Y/IL7B+aPgFp6wiNnThlhVj3tAMAOTj8ssvVyeffHJCERk6dKjab7/91A033KB69uxpbo4LR7GBUlvXcViKstxg/CPyC6wfGn7BKauIDV24ZcWYF4UBvP7662aVamlpiRGUXr16qcGDBzv2iA9XsQGcc25GzqA55wbjH5FfYP3Q8AtuWbnk5ZQV4JYVY14UBtCtWzezSk2aNEl1797dXofRm7POOsuxRwcgM1bJzc1lKzYA3CjTkpvNVfRGbjD+EfmF2bbU4ZS1rqxMFw5IP6YL1rYN3QD69++vjjrqKLNajRs3Th133HH2+rBhw1SPHj0ce3QAImOW6upqFqW8vFwXZ92GLRW23Kwq2up6TJSLmZVyide2lAunrFP69tXFrKdYpB/TLUG0bTqEKjaPPPJIzKiMk/z8/JiRl4cfflhdf/31jj3iw/lQlJMdjjuCb6ysNzdHlnhZqZKobanCKavMsaELt6wY84ZmAIMGDdKHjUyck4Rh4nBWVpZeTlVWRGw6oCg3ibJSJFnbUoRTVhEbunDLijFvaAZgHjpy1ju55JJL1IEHHqjq6upi6hMhYhPLzoYOuSmpSO13iJlkWanRWdtSg1NWERu6cMuKMS85AxCxceOUmw0Rl5vOslIilbalBKes2woLdeGA9GO6YG1bcgYgYhOf8XOKOuSmPLpyk0pWKqTatlTglpVLXk5ZAW5ZMeYlZwAiNompb9xty01RWTRvnJlqVgp0pW0pwCmrHIqiC7esGPOSMwARm+REXW66kjXqdLVtow6nrCI2dOGWFWNecgYgYtM59U0dcpNOGTG9QE1dusV82kDoatYok07bRhlOWUVs6MItK8a85AxAxCY1Glta1We5m9XEhRvVh/M3qHdnF6pR2QUuiUlWZq3Yaj6t76STNaqk27ZRhVNWERu6cMuKMS85AxCxCQZLboImjKxhEVbbhgWnrLNvukkXDkg/pgvWtiVnACI2wWCJzfZdLeYmXwkja1iE1bZhwS0rl7ycsgLcsmLMS84ARGyC4f15xVpsxs0pMjf5ShhZwyKstg0Lblm55OWUFeCWFWNecgYgYhMcYRyOCitrGITZtmHAKavMsaELt6wY85IzABGb4BidvV7ExkfCbNsw4JRVxIYu3LJizEvOAERsgqN8W4MWm9z1VeYm3wgraxiE2bZhwCmriA1duGXFmJecAYjYBEvQh6PCzBo0Ybdt0HDKKmJDF25ZMeYlZwAiNsGyfusOLTZrt2w3N/lCmFmDJuy2DRpuWbnk5ZQV4JYVY15yBiBiEzxBjtqEnTVIMLRtkHDLyiUvp6wAt6wY85IzABGb4LHEpr293dzkOWFnDRIMbRsknLJmX3WVLhyQfkwXrG1LzgBEbILHEpvZ+WXmJs8JO2uQYGjbIOGUVebY0IVbVox5yRmAiE3nZOXkqeN/+lu7nNTrPnOXLrG7tS2ww1FdzRpl0mnbKMMpq4gNXbhlxZiXnAGI2HSOU2qskimW2CwtrDY3eUpXs0aZdNo2ynDKKmJDF25ZMeYlZwAiNsn58LPPXVID5T+fLTB37RIgNEGM2nQla9TpattGHU5ZRWzowi0rxrzkDEDEJjmJxAZK3qrM7vskYuMtXW3bqMMtK5e8nLIC3LJizOupAezZs0ddeeWVasSIEeamwBCx6RyYU2NKjbOccclD5kNSYk5+mRabnQ3+3fG7q1mjTDptG2U4Za1euVIXDkg/pgvWts3YAG688Ub1ySef6OX99ttPXXDBBerrX/+6XRc0IjapccK5d2uJgZ8r1mxQP+rzO5fgrFq3yXxYUuB0bxCb8XOLzU2ekU7WqJJu20YVTlnlUBRduGXFmDdjAzjqqKPsZRAaYNu2beo73/mOXR8kIjbps66o1CU3PXvfr3Y1NJm7JgTEBopfeJU1CnjZtlGAU1YRG7pwy4oxb8YGcPTRR6uGhga9fO2119r13bp1s5eDRMQmc977dJ5LcOYtXmPuFpeC0n23WICffuB1Vsz40baY4ZQVi9hUL1mi5vfvb1a7aNi61axKmUz7cdvex86+4QY1pVcvc1MMK597Tk296CJVPm9eTH17W5v6/J571KSzz1YbPvjArv/k9NNjipO8v/xFt8+uvd8nXSWTrFEj07b1i4wNAObVwCEoKBa/+c1v1EsvveTYKzhEbLwDDkWZgvPjPg+opqbkc2j8HLXxKytG/GxbjHDKikVsgiDTfuyUDlNALKB+d12dXi4aN07Nu+suvbxx4kT1xf33O3eNi/kaIEMAXB265D//sbelQiZZo0ambesX5AxAxMZ7xvxntktwFi0rMHezscSmrHbfSJ6X+J0VE0G0LSY4ZW3cvl0XP4Ev6E/POkvlPvqoXs597DH9JZ/Vp48eBQGcIzawD4xUwOgGLNfm59vPZY3YwL6rhw5VS/Y+l95n+XL1yZlnagGYfN55rucEtHR89Xrme2rZudN+TzAqE4/FjzxiL0/p3VtVGCMygFNMmmtr7XX4uaepSRWMHKlW/eMf9j5Olj31lFr04IP2uvO5ynJy1NSf/cxeTwVO/RjrZ5TnBjBgwAA1Y8YMszowRGz8YfPWKpfc/OD8e9W2HfXmrmrS4k1abEZnJ5afdAkiKxaCalsscMvqd17nF/TWWbNihAC+zAFTbMrnzNHLTdXVatJPfmLtHiM2uX/8o15u2bHDNdIBdCY2FvCe4j3e5MvXX7eX5/XrpwrinHULj92+dq1eBjlzig2UnYWFqnblSpV18cXOh9n7tLW26uXGysqY91G/cWPC95UIv9sVE0H043TI2AD67e1oY8eO1ctwOOr444/XP3P2mm4YiNj4y2czl7gEZ8qspTH7tPh4i4Ugs4ZN0G0bNpyyrhs9Whc/cX4hm3NkLPEwxcZJvMfDvlWLF9v18cQkVbGB55x+ySX2uvn6FvmOaQ0wqlP8/vuOrfuwRmmgwP5Osdn06af2fuZrmOKye9eumHWQJfMxncGpH2P9jMrYAA444AB7+cgjj7SXDzroIHs5SERsgqF4U7lLcE668D61s37f4afxc4q02Lzn8anfYWQNi7DaNiw4ZQ1ijo0pEU4yERt4jEU8sYF9nRN9k4rNpZfa6+brW8R7jWSs+Nvf1JrXXtPLSwcNUisGD7a3mY+H9dbGRledzLFJDayfURkbwLnnnmtfs+b555/XP+EsqRNPPNG5W2CI2ATLx1O/cAnOzPkr1I5dLb6M2oSZNWjCbtug4ZSVstjo5TPOUDk33qhybr45Y7GZe8cdasqFF+q5OKuGDLHr4ZCTRf6LL6p5d96psvr2VdlXXGHXAzAHaM4tt6iZ11yjSj7+2K53zsVxUgPzhvbWz7r+ejV57/dbV+HUj7F+RmVsAE1NTa6zomCezbPPPuvYKzhEbIKnsmaH+t/z9l3wzypw4b8hHy4RsckADG0bJJyyBiE2WJB+TBesbRuaATz44INaQJJJCGzr16+fXVJBxCY85i5a7Rq9+fHFD6q6Ru/eH5asQYCpbYOAU1YRG7pwy4oxrycGkJ+fr/bff3995WH4CeupkkxCkm1LhIhN+JRV1roEB0Z0qmp3mrt2GWxZ/QRj2/oJt6xc8nLKCnDLijFvxgawYcMGfRjqzTffVM3NzfonrEN9KiSTEGtEB8qYMWPMzTYgM1bJzc1N+pzUwNqxgBnzlrsE59yrHjN36xJYs/oB5rb1A25ZueTllBXglhVj3owN4KSTTnKN0MA61KdCqhIC+7W0xL/irVOArFJdXc2ilJeX62LWYyllFZXq++ff6xKcvJVrXfumUjBn9bpgb1uvC6esU/r21cWsD6LAxFizLlmZfccdrrquFLMfw+tDKc7Jidlv/gMPqEk//amadM45av6AAXb9irfe0pORoW76FVe4nh9b4dSPzbb1o6RDalaRhDPPPFNNmzYtpi4rK0vXp0KqYnPIIYeoyZMnm9Uu5FAUPoZ9dXbU5Jm5LsHpdd1Ac/ekYM/qJVFoWy/hlDWIOTarX35ZnyUEV/WtXLhQ18FZTZZYOM9Isij56CP12QUX6OvFwFlLcAaUtb/zDKLZv/61mrxXQtYNG2bXwXZ4DFzYb+2bb9r18foxvHbNsmX2OpyJ5LwgINzXqXbFCr3sfN35d90V9wJ9mDCzUiZe22IgYwPYuXOnPvR022236SsO33rrrXq97qv7dnSGKSE9e/bUPzdt2qTPuLIw90uEiA0+LLGZt7pcr8MVi03B2VJWYzwqPtizekkU2tZLOGUNQmyaqqrsZZAVi3inOAMNZWVq2i9/GbMOOC+2B8Dp1NYtGZb86U/2VYzheRc99NC+fa66Sq159VW9HK8fm2IDp3Evevhhex1ucQC3bgCc18Qpevddfeo2ZsyslInXthjwxADq6+v1CA0IDfyE9c6AKxM7Dx2VlJToektKampqVJ+9/2kceuih6thjj1VtX10wqTNEbPDx5ebtrmvafJG3ziU3Z1yy70MxGdizekkU2tZLOGUNQmxatm/XozZwvRe4FYFFIrGBq+7CfZx2lZbG1JtiYz7eWk9UH68fm2Kz/K9/VUufeMJeXzpwoFr+1SVDnL8nuA4NSBNmzKyUide2GPDFAEBK4AypMBCxwYklNvVNse/1n6M/cwnO6oJNMfs4iUJWr4hK23oFp6xB3FIBJKWuuFi1t7frQ0wWpoA4gTtkg2TAPjvXr9d18cQGZMlZrHpzPyBePzbFpnj8eH0xP4ucm26yb53gfN5VL72kb6CJGTMrZeK1LQZ8MQARm+DA2rFMFhdUuUZtnMxZuMolOGf/6g/mbpHI6hVRaVuv4JbV77wgB8Ce5uYYOTAFxALuiwR3wgZg7ox1t23rUJMFSAmM7lhYV/OF57XuxA1X+YXRIiBeVlNsAHg8jBZBiXm/Z5yhKj7/3N4HO2ZWysRrWwz4YgAiNsGBtWPFI5nYWPTsfb9LcNZv6LgcfFSyekGU2tYLOGUN4lAU3CMJJuTC/ZKcN4KEuTMwmmNOHoY7ei984AE9KXjlc8/FbIM6p1TAbQ7gOebcemvMJF+QEpj4u/Zf/7L3Nfsx7OcsFnB/Jpg/A88Jo0xO4HYKkMUpVFjh1I/NtsWCLwYgYhMcWDtWPEbOKOhUbID8tRtdcvPjPg+opqaWyGT1gii1rRdwyhqE2ARNotEU6cd0wdq2GRmAdY+oeEXEJhiwdqx4bK7apcVmeXFqZ0CN/nCmS3AWLPnS3I0sUWpbL+CUVcSGLtyyYsxLzgBEbHBjHY4qrUl9SHnpykKX4Jz68wGqZXeruSspota2mcIpK0WxSYT0Y7pgbVtyBiBig5vybQ223MxZte+6Nqny1rvTXIKzfHVqt+6IIlFr20zhlFXEhi7csmLMS84ARGzw89HnG2y5GZdTZG5OyrqiLS65OenC+9TO+gZz18gTxbbNBG5ZueTllBXglhVjXnIGIGITDfKKqm25Wbiu0tycECvrR1MWuASnz41/NvaONlFt23ThljVKebdMnarn0FQuWGBu6pSoZc0Ublkx5iVnACI20WLKks224Gys7PyK1WbWtYXuEZwfXnS/amhsjtkvikS9bbsKp6xRORS1a/NmLTRlOTnmppSRfkwXrG1LzgBEbKLHiOn7TgOHsrs1+a0zEmUdN3GuS3B+edvT5m6RgkLbdgVOWaMgNnNuu01LTdWiReamLiH9mC5Y25acAYjYRJNPFm605aasNvF8mc6ymnLz/fPvUdW1O83dIgGVtk0VTlmxiw0IzaxrrzWr00L6MV2wti05AxCxiS75JbW23Fh3AjdJJev0uctdgnPe1Y+Zu6GHUtumAqesWMWmeskSLTU7CgrMTWkj/ZguWNuWnAGI2ESbltY2W26gmHQ1qyk4J15wr9q+I/Vr6IQJtbbtDG5ZMeWFO2uD0Gz48ENzU8Zgy+o33LJizEvOAERsaDAqu2PeTfPuPXZ9OlknZee6BOfC6waau6GDatsmgltWLHkXPfigmtK7t+v+TF6BKWsQcMuKMS85AxCxocPk3I4zpjZV7TtjKpOsMN/GFJwtZand3iEMKLdtPDhlXTlkiC5hsmvvZyWM0mRdfLG5yVOkH9MFa9uSMwARG1qMmN5xWGrWiq0ZZ12wdK1Lbs689GHV2toxKoQF6m1rwilr2HNs5t5++77r0ixcaG7yHOnHdMHatuQMQMSGHq17YufdQIErFn/+ZYXa1ZT+/aL+97y7YwTnhHPvVhVV283dQoND2zrhlDUssVn98staaPa0tJibfEP6MV2wti05AxCxocs7M9e7BMcsM5aVqvVbd6iWFEdgcr5YFSM3UM7+1R/Unj3Jr6cTBJzaFuCUNQyxAaGZ9JOfqLbW9P8ZSAfpx3TB2rbkDEDEhjbNe//ThCsUz11VrsakIDpQRu/dLye/zHyqGF56a4JLcApL9j3mw88+1yVouLUtp6xBik3ztm37znj64ANzUyBIP6YL1rYlZwAiNrRJlBUOVxWW7Yy5wWa8kuzKxiu/LHHJjbOc1Os+8yG+Im1Ll6DadtFDD2mp2Tx5srkpMILKigVuWTHmJWcAIja0STdrW3u7LTdwSKsz/vbqhy6xcZaf/XqQemXEJD0ZuWV3/KH9PW1terQod32VuSklpG3p4nfbFo0dq4WmZWf4V932Oys2uGXFmJecAYjY0CaTrMXldTGjN51hykyq5Ud9fqd+9+RINeDFCeql93P1a6Vyg08TaVu6+HUoqm3v73DS2WerT844w9wUGtKP6YK1bckZgIgNbTLNun1Xiy02WUu3mJtjMIXFq3LJ7U+rt96dpvJWFam2tsSHxqRt6eKH2Kx+5RU9SvPl66+bm0JF+jFdsLYtOQMQsaGNF1mdp4+Pn1NkbraBOTWmlKxYs8HeXl65TX06Y7H684tj1ckX/961bzoFrqnzzMvvq6ycvL3PX+tJ3qjAKavXYrNi8OBQznhKBfmMogvWtiVnACI2tPEy68gZHbdtWFac+ArE//lsgS7JmJNflvQwF9yf6u8jstWvH35LnXvdn11Ck06BQ179Hn5FTZy2UJWWJ37/UcHLtsWOF2IDt0CY+rOf6VGahq1bzc1okM8o/9lcVq1Gvp+tfwYF/JNnfRbd9cir5uZQIWcAIja08Trr+/OKbRmBG3Cmw7otO1IaAQK21jTY+47OXm/Xw4fE8PHT1d2Pve4SmHQLHPL6y0vj1NL85Ie8sOB122KmZNIkXdIFpAaEJvfRR81N6JDPKH8xR5ZhtNdvnFJjlaDPGk0GOQMQsaGNH1mzl5faslFb12xuTsoHDjFatXGbuTku8KVkPQYKzPuJh9W2zkNev7ztadcHSjoFDnnd8/gbaurspaq6NvwzZwA/2hYrmfzdFo0bh+aMp1TIJGsU8TtrQXGp/rt9bdRk9eBTw11/22EWLJAzABEb2viZdZhDNjq703H1ziZ73xHTC8zNKbG8uMZ+Drj+jkk6bQvve/rc5fp09St/83fXB086xTrk9frozxLeNNSLYemuZo0y6RyKmn3DDVpoapYvNzehJp1+HGWSZS2rrFXzF69R/xj2ibp/0Fv6HxVzxCXKBQvkDEDEhjZ+Zt3Z0HHGFJREOEd4Zq7IbG5DU0trzGvCxGYLP9rWOuQFozVnXPKQ64MpnXLh9U+46tIZlvY6K2a6KjbZV16pcm680ayOBH70YyzAId7iTeX6n4m/v/ah+s0fXlMXXjfQ9ffgV/lx3wdcdVCGjZuuPpm2SE3OztWHprLnrVCzF+SruYtWqy/y1qnFywv0Ier8tRvV6oJNal1Rqb7S+sYtlfqfFxglrqzZoecG7qxvUA2Nzaq5Zbd9SBv+eTFfEwoWyBmAiA1t/M7qPGMKbrTpBEZDnHcbr9nZFLM9E6znhGIRdNvC/bHgkBfMy/HqkBcUuLnoz295Up/tBbemWLVuo9od5+ydILOGTapiUzF/vh6lmXfnneamyBB0P86Upubd+st+9Icz9eHfm383RP3kV4+6+rVfBV4LRnNgVOeT6Yv0e0mG+fh0/qlIB3OkKYi5PalCzgBEbGgTVNbhX0nG8OkdogHrVlm7xdu7gIM0wWtZzw9zfbC1LfwnB4ej7njoFfX/fvY71wdqpuXUnw9QN97/knpq6Htq1ucr1daKWvMtkCGe2JiHP7P69EF/xlMqZNKP4SwfL+7TBiMPS1cWqn/9e6r6w7Oj1NW/Hezqf34WeD14XXh9GN2B9+MVMAoL/zzA68DPIHn/03m6YCM0Axg+fLhau3ZtUgnZb7/91NSpU/Vysv2ciNjQJsisH85333eqoHSHuZunmBOLqz38APSLRMPSy1dvUO/t/dB7csg4dcN9L7q2e1kuuuEJdd/Af6lXR07WXxyNTV2bBB40pti0t7VpiQEW3HuvvUyBdD6j4p118/3z79GTZX9157O+iHW88r/n3a363vQXfegWDuHC4ZzNW6tcEuqkq1mjTDptGwShG0AiCWlpaYnZ1qtXLzV48GDHHvERsaFN0FlnrdhqS4Zz/oufNO/eEyM3USDdYWm4zxYc5/9g0nw9/A5fIuaXi5cF5j888cK76t2Pc/R/8LsavDuc2BXMv9usvn1V3pNPaqGZfskljj2jy466Bj1vI2v2UjXmP7PUyyM+VQOf/7c+tHPxzX/RI3Rm+wRRQIpAjp4e+p7uBwvz1qkqD88MDPozKkzMfoyF0A0gkYRMmjRJde/e3V4fOnSoOuussxx7dAAyY5Xc3NyEz0kRrB3LL8LICnNpGpvdc0L8ZnR2xwUEo0AqFzJMRrK2rajarnK+WKWH8gc8OUyfrm5+YXlVeva+Xx86+NNz/977hTxbbdhckfQ/9HRw/t3WbdighUaf8bRsmbFnsEBOmDQKk0s/m7lEjfpgpnr+jY/1aNh19zyveu0Vwx9edL/rdxZ2gf5ww30vqBf+9bH6aMoCPeJTvys8aeUC1u+f0A0gkYSMGzdOHXfccfb6sGHDVI8ePRx7dADPYZbq6moWpby8XBeznmrhlnXs7PW23FRWVbn2oVS8aNt16zeoydMXqD+/MEbd+H8vqDM9OvMrXoFRqktvf1o9M3ScGj9hllqybI3r/cQrtz7wknrn9LN1mbNgmS01zlI4bZrrccnK1rJytXDpKvVZ9kL1zgfT1ZC3PlIPPvm2uu33Q9Qltz2lzr78Edf7x1p6XPh/rjqrmLkxFi/6cVRKEN8/6RDfKgIkkdjIoajUwGrMfsEtK5Ts5R2Hw+Jd74YKYbYtnMYKFz6Ds1Bgkufl/f6qfnD+va4vVq+KJTZmPdYCZ+rA4RuYT/X44DHqzXez9MjIvMVr9KnCcCpwItL9jEr38GbYpJM1qqTbtn4TugGYEtKzZ097GSYPZ2Vl6WVzv0SI2NCGW1Yrb15htS03VTvCGWL3myi0LRyqgTkZcIjmsb+/o668628uCUilBCE2cEgNDh1de/dz+lASHFKC9w3XNlm1bpM+5BTErTYy+YyCs6IyObwZBulmjSKZtK2fhGYAOTk5MYeOSkpKdL0pJZdccok68MADVV1dXUx9IkRsaMMtqzPvhoo6W26Kyryb7IgFSm0LFzWDeSr/ePsT1f+P/1TnX/N4jHQkExuYVAuTqOFxMNkWJt2OnTBHLVjypZ6Mu2Mn/jPlnJj9mDrcsmLMS84ARGxowy2rmRcmMltys2R9esefsWJmpYglLy+e2UsXWIar1VImXj+mDLesGPOSMwARG9pwyxovb0tr9E4HT4V4WSkCImMJDpy9Q51E/Zgq3LJizEvOAERsaMMta6K8bY4L+Y2fW2xujiSJslIk9/HHdeFAsn5MEW5ZMeYlZwAiNrThlrWzvJbcwDVvok5nWSlhXnmYMqn0Y0pwy4oxLzkDELGhDbesqeSlclgqlaxUELGhC7esGPOSMwARG9pwy5pq3o8XlNhyE9StH7wm1awUELGhC7esGPOSMwARG9pwy9qVvNnLS225aQjhFhCZ0pWsUUfEhi7csmLMS84ARGxowy1rV/MudVzIr76pa48Nm65mjTLptG1U4ZQV4JYVY15yBiBiQxtuWdPJO2J6tG6eaZFO1qiSbttGEU5ZAW5ZMeYlZwAiNrThljXdvO/NLY6c3KSbNYrIoSi6cMuKMS85AxCxoQ23rJnkjdpk4kyyRg0RG7pwy4oxLzkDELGhDbesmeStrWuO1KhNJlmjhogNXbhlxZiXnAGI2NCGW9ZM805avEmLzeyVW81N6Mg0a5QQsaELt6wY85IzABEb2nDL6kVea9Qma+kWcxMqvMgaFcoXLtSFA17146jALSvGvOQMQMSGNtyyepG3pLIuEoekvMgaFbxq2yjAKSvALSvGvOQMQMSGNtyyepV3/JwiLTbLimvMTWjwKmsUkBEbunDLijEvOQMQsaENt6xe5sV+lpSXWbEjc2zowi0rxrzkDEDEhjbcsnqd15KbltY95qbQ8TorZkRs6MItK8a85AxAxIY23LJ6nXdUNt6rEnudFTMiNnThlhVjXnIGIGJDG25Zvc7b3t5ui820PFxnSXmdFTMiNnThlhVjXnIGIGJDG25Z/cpryU3bXtHBgl9ZMeJn22KDU1aAW1aMeckZgIgNbbhl9Stvac0uLTZww0ws+JUVI362LTY4ZQW4ZcWYl5wBiNjQhltWP/N+OH+DlpvFBVXmplDwMys25FAUXbhlxZiXnAGI2NCGW1a/83acJRX+KeB+Z8WEiA1duGXFmJecAYjY0IZbVr/zrt60zZabxpZWc3Og+J0VEyI2dOGWFWNecgYgYkMbblmDyDtm5noUp4AHkRULIjZ04ZYVY15yBiBiQxtuWYPKa4kNnA4eFkFlxUDu44/rwoEg+zEGuGXFmJecAYjY0IZb1qDybq7ed5bUyBmpnyUFp4pvrKxXSwur1fS8UjV8+jo14YuNactRUFkxEGTbhg2nrAC3rBjzkjMAERvacMsaZN7K7Y1aboZ/NXqTaana0WS+RFKCzBo28/v314UDQffjsOGWFWNecgYgYkMbblmDzmvKSbzy/txilb18q1peXKNq65tjRmhqdjbF7AuStLOhxfEKiQkqa1tbu5q/psKVyypjc4rU9GWlOh9c78cPZI4NXbhlxZiXnAGI2NCGW9ag89bWNavZK7eqlSW1qqy2wdycMtYFAK2SyiEuv7MWl9ep0dkdE6XTLSBrcMjt871ylO6d0kVs6MItK8a85AxAxIY23LJGOS+M5HwwrzhGCmAUJBF+ZG1sblVTl2x2yQmU/L3y5gTueA5CBsIyca+4pHpILh1EbOjCLSvGvKEZwMSJE1VeXp5ePuWUU4yt+0hHUERsaMMtK5W8MLJhCoE5ydiLrDDi9NHn+66o7CxwV/Oisp3m7hkBI1rW8y/4ssLcnBQRG7pwy4oxb2gGYMrH+PHjY9YBc59UELGhDbes1PIuWlcZIxyfLNxob0s3a33jbjVuTpFLZqDMW12u59T4RUNzq/1aEx1ZOoNi2yaCU1aAW1aMeUMzAFM+rr/++ph1APaxyvPPP29utgGZsUpubq7ruSmDtWP5BbesVPPm5JfFCMi0vC12VjiNvHxbg1q7ebtauLZSZS3dokY7LiLYWXl/XrFrNMhPnBOR4X2nAuW2NeGUFeCWFWPe0AzAlI9fdDIsC/tXVMQf7nUKkFWqq6tZlPLycl3MeqqFW1bKeQs3lbukJN3yn3lFqmRLhes1girLC0rt9/LFqs2u7WaZ0revLmY9xUK9H5uFW1a/86YDGrGJdyjKSY8ePdQ///lPs9qFHIqiDbesXPKWVNapTxaWqDmrytWKDbWqYnujat69x9wNPU7ZgsNUiZA5NnThlhVj3tAMINHk4Z49e+qfDQ0Nqq6uzq5PVVZEbGjDLSu3vFEHDqM55SYRIjZ04ZYVY95QDeDCCy9UBx98cMz8GUtKWlpa1KWXXqoOP/xw9a1vfUuvp4KIDW24ZeWWlwojphfYchNvvo+IDV24ZcWYl5wBiNjQhltWbnkp8d7cjmv47G6NvZCfiA1duGXFmJecAYjY0IZbVm55qfH5lx1nTL0zc71dz6ltOWUFuGXFmJecAYjY0IZbVm55KQK3crDkxrrycsmkSbpwQPoxXbC2LTkDELGhDbes3PJSZfuuFltusvK2yKEownDLijEvOQMQsaENt6zc8lLGeVuJCb37iNgQhVtWjHnJGYCIDW24ZeWWlwNwp/OPL+yjC0hOQekOcxdSSD+mC9a2JWcAIja04ZaVW14uTO5zsS02VslevtXcjQTSj+mCtW3JGYCIDW24ZeWWlwtW28JNOp1yAyW3oMrcPdJIP6YL1rYlZwAiNrThlpVbXi6YbbuhouPMKaus20LjEJWZlTrcsmLMS84ARGxowy0rt7xcSHZWVEkcyfloQYm5W2SQfkwXrG1LzgBEbGjDLSu3vFxIJjYWKzfUugQne3lp3Fs0YEb6MV2wti05AxCxoQ23rNzyciEVsbGorWvWZ1KZkrNwXaVqakl8B3EsSD+mC9a2JWcAIja04ZaVW14udEVsnNTWN6sP5m9wSY5V4LYNcHVjuGYOFqQf0wVr25IzABEb2nDLyi0vF1YOGaJLJsCNNZcWVqtR2e7RHKt89PkGVVi203xooEg/pgvWtiVnACI2tOGWlVteLvjRtnDbhjn5ZS65cZYJX5So8XOL1Yjp7m1dKXDn8s/XVKhNVfWqrS35nB8/smKGW1aMeckZgIgNbbhl5ZaXC0G07ebqXWry4k0uKfGzDN8rTAu+rFCbq3bZwhNEVkxwy4oxLzkDELGhDbes3PJyId05Ntio2tGk8gqr1ZhZhS7J8avAqBOcMVbfiLO/cOrHWD+jyBmAiA1tuGXllpcLVMQmEZXbG/X8n4lfbHSJSbICIz6jZ65X/55dqMbNKdKHvMx94hXYH0aKWnbvMd9K4HDqx1g/o8gZgIgNbbhl5ZaXC9TFxomX/RhuGDp16RaX2CQqMAcoaLzKGgW8bFsvIWcAIja04ZaVW14uiNh4R2nNLn3PLRjpMcXGKjAStGrjNvOhvuBnVmz43bbpQs4ARGxowy0rt7xc4NS2YWZdv3VHwtPh4f5cfhBW1jAIs22TQc4ARGxowy0rt7xc4NS2WLJWbGt0yY1VVmyoNXdPGwxZgwJL25qQMwARG9pwy8otLxfkUFS4fLl5u0tuoIzNKdSnqmcCtqx+grFtAXIGIGJDG25ZueXlgogNDva0tesLDZqCk0qB21fAmVtwiws4Bf2z3M1q+rJSNXN5qZqzqkzP+4HnhrO1Fq2rVIsLqtSS9dX69Hi47QWcsp5fUqtWb9ymRWtd6Q5VVB7uVaK7Cta2JWcAIja04ZaVW14uiNjgBc68SjQvJ+gyYnqByiuqMd+iZ1TvbNLiNTu/TMtVS2vXTpfH2rbkDEDEhjbcsnLLywURG7p0JWt7e7seNYL7fjXv3qOv/wMjP6bgWGX2yjLzKVKmbe9rFZXt1IfbzOdNVj5ZuFHfSR4mW8P7dYK1bckZgIgNbbhl5ZaXCyI2dPEqa1PLHjU/yWGyiXuFA051jwcIyMbKepW9vNT1OLOMzi5QHya5Y3yyMmbWepW1dLP58qFDzgBEbGjDLSu3vFzg1LacsgJ+ZAVRgevwjM6Of62eMTPXq5z8spRubgr7ba1pMF/CBdyyAuYCTcvbop/ffB5nwQY5AxCxoQ23rNzycmHd6NG6cED6sX/AoSVTMmB+EExcrtjeaO7uOVjblpwBiNjQhltWbnm5IIei6MItK8a85AxAxIY23LJyy8sFERu6cMuKMS85AxCxoQ23rNzyckHEhi7csmLMG5oBTJw4UeXl5enlU045xdi6j0MOOUS9/fbbehlkxTzVLB4iNrThlpVbXi6I2NCFW1aMeUMzAFM+xo8fH7MOOPe5/fbb1T333OPYGh8RG9pwy8otLxc4tS2nrAC3rBjzhmYApnxcf/31Mev5+fkx+0yYMEEdffTRjj06AJmxSm5uruu5KYO1Y/kFt6zc8nKBU9tyygpwy4oxb2gGYMrHL4xh2QULFsTsM2PGDPXNb37TsUcHsJ9ZqqurWZTy8nJdzHqqhVtWbnnNOqplzr336mLWUyzSj+mWINo2HdCIjRyKSg+sxuwX3LJyy8sFmWNDF25ZMeYNzQASTR7u2bOnvQyTh4cNG6aXQVZk8rAbrB3LL7hl5ZaXCyI2dOGWFWPeUA3gwgsvVAcffLB6/vnn7TpTSq699lp9CKqioiKmPhEiNrThlpVbXi6I2NCFW1aMeckZgDV5WIoUKVKkSJES7QKDFV2FnNhwG7FJt+GjiHXmGxekbenCqW05ZQXk+yd8yLWAiA1d5MuPLtK2dOGUFZDvn/Ah1wIiNnSRLz+6SNvShVNWQL5/wodcC4jY0EW+/OgibUsXTlkB+f4JHz4tIAiCIAgCeURsBEEQBEEgg4iNIAiCIAhkELERBEEQBIEM5MSmd+/e+lYMGK+GmCkzZ85UZ511ltp///3VFVdcEbMtOztbHXTQQeqJJ56IqacATFB79tln7XXIetRRR8XcfoMCLS0t6vzzz1fHHXecev/99+36Sy+9lFzbbtq0SWc64YQT1IoVK+x6q22jnnXdunWqX79+cSeSJmrPwsJCfZV1eFzUgPf8jW98I+bvFG5kDP0Z2nPs2LGOvZVaunSp+va3v63bP2oka1vgjTfeUP3794+pu/7669WBBx6oHnjggZh67FhZzbYF2tra1NFHH62OOeYYndnCatsws8ZvmYgCHc26/1SiThdlrrvuOnsZvvjGjBmjl5ctW6Y7HtC3b9+Y/aLOSSedpK6++mr7j8qZtaGhgUzWxsZG3WctIV++fLn+CXmnTp2qlym1rfX3CR+OsNzU1ESqHw8fPlyNGzfO9Tm03377xbSnBfRla9+BAweq7373u/a2KABZ4UbFzi+/Pn36qD179ujlQw89VD3zzDN6eevWrXZW2H722Wfbj4kCidoWgP582GGHxYgN5H377bf18m233RapvFZWs23hvo3O/EuWLNE/nW0bZlZ3y0SU0tLSmF80fDDEu2M4JXr06KF/wgjV7Nmz7fp4f3BR5Y477tD/MVh/VFSzwn8+8B+7CeR1QiFvfX19TA64X9zmzZtJtq2ZwVwfPHiw/nnxxRerJ5980q4394sCzr9Tk88++0yPRgEnnniiGjVqlL0tilmBeO8b+jJ86TvFBvI6ifc47Jhte9ppp+kjCCZY2jacV/WBv//973rY0wKGeX9B+CZz8N8eHLoAzM5jrkcVK4fzj8rMZq5HFcgBpaCgQAvrySefbNeb+1GgsrLSzjxy5EhdZ2Yz16OImQEOMzqx/jmB/Zxiaz4uCphffk4gT01Njb3c2toasy2KmO8bxA1GbACn2Jj7metRwGxb628X/kk555xz1LHHHmvXY2jbcF7VB1577TV1+umn2+sDBgzQxzUpAtn+67/+y143O4+5HkXuvPNO9cEHH+hlLmIzZ86cmHXnT7M+ymzfvl0fkrGwRmrMbOZ6FDEzdO/ePWYd5swBsJ9zrpH5uChgfvlZQJasrKyY9R07dsSsRxHzfb/wwgv2Mgexeeedd2LWrZ8Y2jacV/UB84rDVA9F/fGPf1TdunWLqaM4hA8ZIYezHH/88SSzAvBFD3NMLKxcFA9FweGXM844w16HOVT33XcfybY1M5jr1A9Fvf7662ry5MkxdVgOV2SK+b7NzytrO8VDUYcffriaMGGCve7MiqFtw3lVn4BfIuXJw4MGDXL9xwdAZiqTLuPh/KNyZqU0eRiOV1ujcL///e/1BwcAea3/dqm0bVVVlf33CYdTYbm4uJhkPzY/h0Bgne1p4Zw8DIfRv/Od79jbooL55Tds2DB7Xo2TqE8etjDb1ok5eRh+FwBMwo1iXrNt165dqw444AC9/OKLL9p/t862DTNr4pYRBEEQBEGIGCI2giAIgiCQQcRGEARBEAQyiNgIgiAIgkAGERtBEARBEMggYiMIgiAIAhlEbARBEARBIIOIjSAIgZDsuh/JSPdxgiDwRD4xBEFICEjFiBEj9DJclAvW4T5PqRBlISkpKYn0+xcEzshfriAIcbn22mvVunXrYur++te/xnzhw/L69ev1zR3vvfdeux5ufwHbrGLta/HUU0/pn3369NFXWZ44caJeP++889Shhx5q7wc4H7dt2zZ9O4b//u//jnk94NZbb1UHHnig62qn8Pjhw4erI488Uo0ePTpmG/DQQw/p2znAjfxgGXC+9969e9v7vvHGG/qKq1deeaVdB8B+5eXl+vVvueWWmG2CIASLiI0gCHFJNGIB9dYdfGH51FNP1cvf+973dHHu58QUG7gMO9wwr7S0VG87+OCD9a0FQJQSPQ8sw+0n4FL8b775Zkw9SAeQk5OjzjzzzJhtQ4YM0ctlZWV2vQXs297erm/vYO0Xb8Tmhz/8obrooov08tChQ13vy3rPp5xyijriiCPsbYIgBEv8Ty5BENhjfrFbQP3ixYvtZXNbvGVzHcRm165d9jrcBBLkwiLR85jPaXHNNdfErKfyGIuPP/7YrIorNuY6yMuiRYvibjPXBUEIDvnrEwQhLom+nKHeOWJjbquoqEi4zcI6FGUBN9lzkkxM4C7YUGfdeA94/PHHHXvEYj7eZP78+er888/X+8GoEWCKzcKFC5M+j7kN1i3pEQQhWBL/pQqCwBoYBUlljo2TVLdlIjYWMJ/FOnR09NFHG1s7SPT4eFj7btq0yfU4c92Juc1cFwQhOOSvTxCEhMAX9MiRI/VyQUGBXodJss7tp59+ul6GCcPf/e53Y7YlOryUrtj86U9/Us3NzaqwsFDXr1mzxt7n9ttv18swmmS+j0TAyMyyZcv08sqVK12va41MATB3plu3bnp5+/bt9usBsO8JJ5ygl0877TQ9IVoQhHBI/BcvCAJ7Ghsb1WWXXaZHR3r06GFu1l/oIDxwVtTdd98ds+3pp5/WZxtZsuCF2AwcOFAL1IknnqiWLl1q1wP333+/Ouyww/TozRdffGHXJxOb3bt3q5/+9Kfqm9/8pvrRj36kJy5bvPrqq/r5nGdFvffee3pfkJhXXnnFrofXAOGDbTfffLNdLwhC8CT+ixcEQeiEZNLACfk9CAIe5K9REIS0kS/0fcjvQRDwIH+NgiAIgiCQQcRGEARBEAQyiNgIgiAIgkAGERtBEARBEMggYiMIgiAIAhlEbARBEARBIIOIjSAIgiAIZBCxEQRBEASBDCI2giAIgiCQQcRGEARBEAQyiNgIgiAIgkAGERtBEARBEMggYiMIgiAIAhn+P6JeimpZQNJjAAAAAElFTkSuQmCC>