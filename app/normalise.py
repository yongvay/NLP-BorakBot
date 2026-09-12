"""Repair transcription errors before generation — Stage 2 of the pipeline.

    Whisper transcript  ->  whitespace/punctuation tidy  ->  ASR repair  ->  Stage 3

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


ASR_REPAIRS: dict[str, str] = {
    # Invented tokens -- no legitimate occurrence, so repair is free.
    "jbj":        "myjpj",    
    "charanabuk": "nak",       
    "unitrasca":  "unit",      
    "maham":      "mahal",     
    "mora":       "murah",    
    "chiamnna":   "camne",     
    "naklam":     "nak",       
    "jepya":      "jpn",       
    "bernil":     "renew",     
    "lemala":     "lemak",     
    "identi":     "identity", 
    "autor":      "auto",    
    "ayo":       "aiyo",       
    "efilling":  "efiling",   
    "pasport":   "passport", 
    "license":   "licence",  
    "lho":       "lor",       
}

# Whisper punctuates and capitalises; the corpus does too (386 of 506 inputs end
# in .?!). So punctuation is kept. Only the artefacts are cleaned: repeated
# marks from hesitation, and space before a mark.
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:])") # Find one or more spaces followed by punctuation
_REPEATED_PUNCT = re.compile(r"([,.!?;:])\1+")     # Detects repeated punctuation marks
_WHITESPACE = re.compile(r"\s+")                   # Detects one or more whitespace characters

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
