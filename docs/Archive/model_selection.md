# Base model selection — Llama-3.2-3B-Instruct

Decided 22 August 2026. Supersedes nothing; this is the first binding choice of
base model. `training/qlora_config.yaml` carried a Qwen placeholder until now.

**Winner: `meta-llama/Llama-3.2-3B-Instruct`** (LLaMA-Factory template `llama3`).
Runner-up: `mesolitica/mallam-3b-20k-instructions`.

## How the bake-off was run

Both candidates answered the same committed 20-item probe set
(`eval/probe_set.jsonl`) un-fine-tuned, 4-bit, greedy decoding, 160 new tokens,
under the production system prompt. Raw generations are committed at
`eval/results/llama.json` and `eval/results/mallam.json`; the aggregate is
`eval/results/refusal_report.csv`.

MaLLaM was run twice. Round 1 is kept at
`eval/results/mallam_round1_no_template.json` as evidence of a failure mode, not
as a result: its tokenizer ships no `chat_template`, the harness fell back to a
plain prompt, and all 20 replies were degenerate. Round 2 forces
`--chat-format mistral` and is the run scored here. The same hazard applies to
training, which is why the template field below is called out.

## Why Llama won

The probe set has 17 answerable items and 3 designed out-of-knowledge items. The
two models fail in opposite directions:

| | Llama-3.2-3B | MaLLaM-3B |
|---|---|---|
| Refusal probes correctly declined | **3 / 3** | **1 / 3** |
| Answerable items wrongly refused | 14 / 17 | 3 / 17 |
| System prompt leaked into the reply | 0 | 2 |
| Truncated mid-sentence | 1 | 4 |
| Degenerate repetition loop | 1 | 0 |

MaLLaM reads better. Its sentences are well-formed Malay, and on a fluency-only
impression it wins easily. That impression is what this document exists to
overrule, because the fluent sentences are frequently false.

**MaLLaM fabricates on the graded axis.** CLAUDE.md makes hallucination control a
requirement, and singles out live/real-time factual queries as items that must
hit the fallback. Asked `eh ada peraturan JPJ baru ke minggu ni?`, MaLLaM invents
a regulation:

> Ya, terdapat peraturan baharu yang diperkenalkan oleh Jabatan Pengangkutan
> Jalan (JPJ) berkaitan dengan penggunaan kenderaan di jalan raya...

Asked `boleh saman company saya tak kalau kena fire?` it answers
`Boleh, tiada masalah. Anda boleh saman syarikat anda jika anda tidak tahu.` —
incoherent, and legal advice. Llama declined both.

The pattern repeats on in-domain items where it is confidently wrong rather than
absent: `2 hingga 3 hari bekerja` for a Sarawak processing time whose gold answer
is 7 working days; `Kelas A. RM30.` where the gold answer is classes D, E, F, G,
H and I; `kiasu` glossed as `sombong atau sombong`.

**Llama's failure is the correctable one.** Its 14/17 over-refusal looks worse in
the aggregate and is the reason it reads badly, but it is an un-fine-tuned base
with no JPJ/JPN knowledge baked in. Declining a question it cannot know is the
designed behaviour, and the knowledge that removes the need to decline arrives at
fine-tuning time. Training a model to answer facts it has been given is a
routine outcome of the QLoRA run; training a model out of confident fabrication
is not.

Over-refusal is also the safe direction to be wrong in for a graded demo. A bot
that says `tak pasti` too often loses marks on helpfulness. A bot that invents
JPJ regulations on stage loses marks on the requirement the brief calls out.

**Instruction-following, not just register.** Llama's over-refusal is itself
evidence that it obeys the system prompt — it followed the fallback instruction
past the point of usefulness. MaLLaM ignored that instruction and twice emitted
the prompt's own bullet list as its answer. Fine-tuning has more to work with in
a model that already follows instructions.

**Register was not the deciding factor, and did not favour MaLLaM.** The target
is Bahasa Rojak. MaLLaM defaults to formal BM baku (`sila hubungi pihak
berkenaan`); Llama produced particle-bearing rojak unprompted (`Sama-sama lah!
Apa yang diinginkan, kawan?`). Being Malay-native biases MaLLaM toward formal
Malay, which is further from the target register, not closer.

## Why there are no human Likert ratings behind this

`docs/DESIGN.md` §7 originally specified a blinded Likert pass as the selection
instrument. That pass was not run, and the decision rests on the refusal counts
and the qualitative failure modes above instead.

The gap between the two candidates on hallucination (3/3 vs 1/3, with two
fabricated answers quotable verbatim) is categorical rather than marginal, and a
subjective 1-5 mean from the two people who chose the candidates would have been
the weaker evidence of the two. The blinded sheet is still generated and
committed at `eval/rating_sheet.csv` for anyone who wants to check the call.

This does **not** discharge the human evaluation the assignment requires. That
one compares the fine-tuned model against the un-fine-tuned base with the
tokenizer held constant, is a different comparison from this one, and is still
outstanding.

## What this pins

- `training/qlora_config.yaml`: `model_name_or_path: meta-llama/Llama-3.2-3B-Instruct`,
  `template: llama3`.
- Llama-3.2 is a gated repo. The HF token must be one that has accepted Meta's
  licence, or the Colab run fails at model load rather than at training.
- Unlike MaLLaM, Llama-3.2-3B-Instruct ships a `chat_template` in its tokenizer
  config, so the round-1 failure mode does not apply. Verify anyway before the
  training run — a wrong template does not raise, it trains on mis-delimited text:

      python -c "from llamafactory.data.template import TEMPLATES; print('llama3' in TEMPLATES)"
