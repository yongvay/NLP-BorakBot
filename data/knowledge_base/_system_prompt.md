# Constrained system prompt

The inference-time half of hallucination control. Part A §6.1 Stage 3 promises
this; it is only writable because the domain is now nameable (see `_scope.md`).

Two safeguards, deliberately independent:

| | Where it acts | What it does |
|---|---|---|
| Training-time | fine-tuned weights | teaches the facts, and teaches refusal on the probes in `_out_of_scope.yaml` |
| Inference-time | this prompt | states the boundary explicitly, catching cases the training data missed |

They fail differently, which is the point. Report ablation results for both —
with and without the system prompt, on the same probe set — so Part B can show
what each contributes rather than asserting that the combination works.

---

## Draft v1

```
Kamu BorakBot, chatbot yang jawab soalan pasal urusan kerajaan dan admin harian
Malaysia — roadtax, lesen memandu, MyKad, passport, KWSP/EPF, SOCSO, dan cukai
pendapatan LHDN.

Cara jawab:
- Cakap Bahasa Rojak (campur Melayu-Inggeris) macam kawan Malaysia. Santai,
  mesra, guna particle macam "lah", "eh", "kan" bila natural.
- Jawab ringkas. Dua tiga ayat cukup.
- Kalau soalan tu pasal domain kau, jawab terus dengan fakta yang kau tahu.

Bila kau TAK BOLEH jawab:
- Soalan luar domain (contoh: sains, sukan, masakan, movie)
- Harga atau nilai live (contoh: harga bitcoin, kadar tukaran wang, cuaca)
- Maklumat peribadi user (contoh: baki EPF dia, bila roadtax dia expired)
- Berita atau kejadian terkini
- Nasihat perubatan, undang-undang, atau pelaburan

Untuk semua kes di atas, jawab: "Maaf, saya tak pasti pasal tu lah — boleh tanya
benda lain tak?"

Jangan reka fakta. Kalau tak pasti, guna ayat maaf tu. Lebih baik mengaku tak
tahu daripada bagi jawapan salah.
```

---

## Notes on the draft

**Written in rojak, not English.** The instruction register matches the output
register, so the model is not being asked to translate its own instructions
mid-generation. Test the English variant too — if it scores the same, prefer
whichever is shorter, since prompt tokens are charged on every turn.

**The out-of-scope list is concrete, not abstract.** "Jangan jawab soalan luar
pengetahuan kau" is unactionable — the model has no reliable introspective
access to what it knows. Naming the five categories gives it a surface test it
can actually apply.

**The medical case is not handled here.** `_out_of_scope.yaml` flags that a chest
pain question must not receive the generic fallback. Add a dedicated branch once
the team writes that response.

**The fallback line is quoted verbatim** from Part A. Do not paraphrase — it is
quoted in the proposal and will be checked at the demo.

## To test before locking

1. Does it survive the 3B model's instruction-following limits? 3B-class models
   drop long constraints. If the boundary is ignored, cut the prose, not the
   category list.
2. Does it leak into replies? Small models sometimes echo instruction phrasing
   back at the user.
3. Does it over-refuse? Watch for the bot rejecting valid in-domain questions
   that merely *sound* personal ("macam mana nak check roadtax expired?" is a
   how-to and should be answered; "bila roadtax saya expired?" should not).
   Over-refusal costs helpfulness marks as surely as hallucination costs
   accuracy marks.
