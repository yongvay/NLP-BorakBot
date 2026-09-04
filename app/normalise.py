"""Repair transcription errors before generation — Stage 2 of the pipeline.

    Whisper transcript  ->  whitespace/punctuation tidy  ->  ASR repair  ->  Stage 3

WHAT THIS DOES NOT DO, AND WHY

Part A specified Malaya's informal-Malay normaliser here ("xleh" -> "tidak
boleh") plus a slang dictionary mapping Manglish to canonical forms. **That was
the right design for an un-fine-tuned model and is the wrong one now.**

371 of the 506 training inputs (73%) are informal rojak -- "sapa reka telefon
wei", "lepak tu maksud apa actually?". `training/to_llamafactory.py` passes the
`user` field through verbatim, so the adapter was fitted on exactly that
register. Expanding "xleh" to "tidak boleh" before generation would hand the
model text unlike anything it was trained on, and the fine-tuning bought a 6.5x
perplexity improvement that is only collectable on-distribution. Formalising the
input spends it.

Nor does this lowercase, and nor does it strip punctuation. Capitals in the
training inputs are load-bearing -- `MyKad` (14), `JPJ` (10), `P` and `D` as
licence classes (12 together), `IC` (6) -- and 391 of 506 inputs contain a
question mark. Case-folding "P" to "p" would erase the difference between a
probationary licence and a letter of the alphabet.

So Stage 2 keeps the register and fixes only what the *speech* stage got wrong.
DESIGN.md §5 measured those errors and named this as the remedy: `myjpj` heard
as "jbj" 9 times, `aiyo` as "ayo" 9 times. Deviation from Part A §5.2; reasoning
recorded in DESIGN.md §12.

THE RULE FOR ADDING AN ENTRY

Only repair a token that **cannot legitimately occur in Malaysian rojak**. The
substitution table (`stage1_stt/results/substitutions.csv`, 492 rows)
is tempting and mostly unusable, because it is keyed the wrong way round: it
records what each reference word turned into, not what each mistake came from.
Reversing it blindly is destructive.

    Whisper wrote "saya" for `can` (9), `do` (6), `i` (6), `balance` (5), `my` (4)
    Whisper wrote "card" for `roadtax` (6) and `mykad` (6)
    Whisper wrote "buku" for `pukul` (8) -- and "buku" is also just the word for book

None of those can be repaired by lookup. What can be repaired is the residue
Whisper invents that is not a word in either language: "jbj", "chiamnna",
"unitrasca". Those are safe because a correct transcript never contains them.

    from app.normalise import normalise
    result = normalise("eh jbj ni buka pukul berapa")
    result.text      -> "eh myjpj ni buka pukul berapa"
    result.repairs   -> [("jbj", "myjpj")]

Usage
    python app/normalise.py "eh jbj ni buka pukul berapa"
    python app/normalise.py --self-test
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field

# ------------------------------------------------------------------- the table
# Every entry is a token Whisper produced that is not a word in Malay or English,
# or a spelling of a domain term that the training corpus never uses. Counts are
# occurrences in the 48-clip evaluation set; see the module docstring for the
# rule and stage1_stt/results/substitutions.csv for the raw data.
#
# Deliberately NOT included, though they appear in that file with high counts:
# nampak->nak, look->dulu, good->kot, and->n, dengan->money, kad->credit. Each
# of those is a real word being mistranslated, and repairing it would corrupt
# the many utterances where the word is correct.

ASR_REPAIRS: dict[str, str] = {
    # Invented tokens -- no legitimate occurrence, so repair is free.
    "jbj":        "myjpj",     # 9
    "charanabuk": "nak",       # 6
    "unitrasca":  "unit",      # 6
    "maham":      "mahal",     # 6
    "mora":       "murah",     # 6
    "chiamnna":   "camne",     # 4
    "naklam":     "nak",       # 3
    "jepya":      "jpn",       # 3
    "bernil":     "renew",     # 3
    "lemala":     "lemak",     # 3
    "identi":     "identity",  # 3, truncation
    "autor":      "auto",      # 3

    # Orthography. Same word, other spelling; the corpus uses the right-hand
    # form. These overlap stage1_stt/orthography_map.json, which is the
    # lenient-WER map -- kept separate because that file scores a frozen result
    # and this one feeds a live model.
    "ayo":       "aiyo",       # 9
    "efilling":  "efiling",    # 7
    "pasport":   "passport",   # 6
    "license":   "licence",    # 5
    "lho":       "lor",        # 3, discourse particle
}

# Whisper punctuates and capitalises; the corpus does too (386 of 506 inputs end
# in .?!). So punctuation is kept. Only the artefacts are cleaned: repeated
# marks from hesitation, and space before a mark.
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:])")
_REPEATED_PUNCT = re.compile(r"([,.!?;:])\1+")
_WHITESPACE = re.compile(r"\s+")

# Split on word boundaries but keep the separators, so punctuation and spacing
# survive reassembly untouched.
_TOKENS = re.compile(r"(\w+|\W+)")


@dataclass
class Normalised:
    text: str
    repairs: list[tuple[str, str]] = field(default_factory=list)
    changed: bool = False


def tidy(text: str) -> str:
    """Whitespace and punctuation artefacts. Case and wording are left alone."""
    text = _WHITESPACE.sub(" ", text).strip()
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _REPEATED_PUNCT.sub(r"\1", text)
    return text


def _match_case(original: str, replacement: str) -> str:
    """Carry the original's capitalisation onto the repair.

    Whisper capitalises the first word of an utterance, so "Jbj" and "jbj" both
    occur and both must map to the same repair without shouting.
    """
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def repair(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Token-level lookup against ASR_REPAIRS. Returns (text, what changed)."""
    out, changes = [], []
    for tok in _TOKENS.findall(text):
        fixed = ASR_REPAIRS.get(tok.lower())
        if fixed is not None:
            fixed = _match_case(tok, fixed)
            changes.append((tok, fixed))
            out.append(fixed)
        else:
            out.append(tok)
    return "".join(out), changes


def normalise(text: str) -> Normalised:
    """Stage 2. Tidy, then repair. Safe on already-clean typed input."""
    tidied = tidy(text)
    fixed, changes = repair(tidied)
    return Normalised(text=fixed, repairs=changes, changed=fixed != text.strip())


# ------------------------------------------------------------------- self-test
# Cheap enough to run at the command line before a demo, and it documents the
# boundary of what this stage is willing to touch.

_CASES = [
    # (input, expected output)
    ("eh jbj ni buka pukul berapa",      "eh myjpj ni buka pukul berapa"),
    ("Jbj tutup ke?",                    "Myjpj tutup ke?"),
    ("ayo lupa bawa ic  lah !!",         "aiyo lupa bawa ic lah!"),
    ("nak renew lesen  ,  boleh?",       "nak renew lesen, boleh?"),
    # Register is preserved: no expansion, no case-folding, no particle removal.
    ("xleh renew online ke bro",         "xleh renew online ke bro"),
    ("berapa lama kena pakai P ni?",     "berapa lama kena pakai P ni?"),
    ("MyKad hilang macam mana",          "MyKad hilang macam mana"),
    # Ambiguous forms are left alone on purpose -- see the module docstring.
    ("aku beli buku semalam",            "aku beli buku semalam"),
    ("dia nampak sedih",                 "dia nampak sedih"),
    ("card aku hilang",                  "card aku hilang"),
]


def self_test() -> int:
    failed = 0
    for src, want in _CASES:
        got = normalise(src).text
        ok = got == want
        failed += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {src!r}")
        if not ok:
            print(f"         want {want!r}")
            print(f"         got  {got!r}")
    print(f"\n{len(_CASES) - failed}/{len(_CASES)} passed")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Repair ASR errors in a rojak transcript.")
    ap.add_argument("text", nargs="?", help="transcript to normalise")
    ap.add_argument("--self-test", action="store_true", help="run the built-in cases")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.text:
        ap.print_help()
        return 1

    result = normalise(args.text)
    print(result.text)
    if result.repairs:
        for before, after in result.repairs:
            print(f"  repaired: {before} -> {after}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
