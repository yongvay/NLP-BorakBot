#!/usr/bin/env python3
"""Build the Stage 2-4 half of the Part B report as Markdown.

    committed evidence  ->  this script  ->  docs/PartB_Stages2-4.md

Every figure in the output is read from `eval/results/*.csv`,
`finetune/.../trainer_state.json`, the corpus and the knowledge base at build
time. Nothing is transcribed by hand, so the report and the evidence cannot
drift, and "where does this number come from" has a one-word answer.

Markdown rather than .docx on purpose. The Stage 1 draft styles its tables with
direct formatting rather than a named style, so appending programmatically means
cloning table properties and moving XML nodes -- a lot of fragile machinery to
produce a file that still has to be opened and checked in Word. Emitting
Markdown lets the content be reviewed on its own and pasted in with Word's own
styles applied, which is both faster and easier to correct.

The prose lives in `report_sections.py` and `report_backmatter.py`. They call an
emitter object; this module is that emitter. Splitting them keeps the wording a
marker reads out of the string plumbing.

Usage
    python scripts/build_report.py
    python scripts/build_report.py --check     # is the committed file current?
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "PartB_Stages2-4.md"

# The Stage 1 draft ends at Table 6 once its captions are renumbered -- they
# currently run 1, 2, 3, 5, 6, 4, 5. The output says so in its header.
FIRST_TABLE = 7


# --------------------------------------------------------------------- evidence

def evidence() -> dict:
    e: dict = {}

    rows = [json.loads(l) for l in (ROOT / "data/generated/pairs_v1.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    know = [r for r in rows if "domain" in r]
    ref = [r for r in rows if "source_probe" in r]
    e["pairs_total"], e["pairs_know"], e["pairs_ref"] = len(rows), len(know), len(ref)
    e["n_facts"] = len({r["source_fact"] for r in know})
    e["n_probes"] = len({r["source_probe"] for r in ref})

    e["domains"] = {}
    for r in know:
        e["domains"][r["domain"]] = e["domains"].get(r["domain"], 0) + 1
    e["refusal_cats"] = {}
    for r in ref:
        e["refusal_cats"][r["category"]] = e["refusal_cats"].get(r["category"], 0) + 1

    for s in ("train", "val", "test"):
        e[f"n_{s}"] = sum(1 for l in (ROOT / f"data/generated/splits/{s}.jsonl")
                          .read_text(encoding="utf-8").splitlines() if l.strip())

    e["scores"] = {r["tag"]: r for r in csv.DictReader(
        (ROOT / "eval/results/score_report.csv").open(encoding="utf-8"))}
    e["refusal"] = {r["tag"]: r for r in csv.DictReader(
        (ROOT / "eval/results/refusal_report.csv").open(encoding="utf-8"))}

    state = json.loads((ROOT / "finetune/borakbot-qlora-r1/trainer_state.json")
                       .read_text(encoding="utf-8"))
    e["evals"] = [(h["step"], h["epoch"], h["eval_loss"])
                  for h in state["log_history"] if "eval_loss" in h]
    e["max_steps"] = state["max_steps"]
    allr = json.loads((ROOT / "finetune/borakbot-qlora-r1/all_results.json")
                      .read_text(encoding="utf-8"))
    e["train_loss"] = allr["train_loss"]
    e["train_runtime"] = allr["train_runtime"]

    sys.path.insert(0, str(ROOT / "app"))
    sys.path.insert(0, str(ROOT / "eval"))
    from normalise import ASR_REPAIRS                    # noqa: E402
    from inference import load_system_prompt             # noqa: E402
    e["repairs"] = dict(ASR_REPAIRS)
    e["system_prompt"] = load_system_prompt()

    # Per-domain figures, computed by the same function section 14 quotes.
    import score                                          # noqa: E402
    per = {}
    for tag in ("base", "tuned"):
        payload = json.loads((ROOT / f"eval/results/{tag}.json").read_text(encoding="utf-8"))
        per[tag] = {d["stratum"]: d for d in score.by_stratum(payload)}
    e["by_stratum"] = [
        (k, per["base"][k]["n"],
         f"{per['base'][k]['bleu']:.2f}", f"{per['tuned'][k]['bleu']:.2f}",
         f"{per['base'][k]['rougeL']:.2f}", f"{per['tuned'][k]['rougeL']:.2f}")
        for k in sorted(per["base"], key=lambda k: -per["tuned"][k]["bleu"])
    ]
    return e


# ---------------------------------------------------------------------- emitter

class Markdown:
    """The interface the content modules call.

    `at(...)` exists because the content was written against a document-insertion
    API. Here anchors are meaningless -- everything appends in call order -- so it
    returns a sentinel the other methods ignore. `replace(...)` collects the
    paragraphs that must be edited in the existing Stage 1 draft; those are
    emitted together at the end rather than pretending to be new content.
    """

    def __init__(self) -> None:
        self.out: list[str] = []
        self.edits: list[tuple[str, str]] = []
        self.table_n = FIRST_TABLE - 1

    def at(self, _pred=None):
        return None

    def para(self, _at, text: str, bold_lead: bool = False) -> None:
        text = " ".join(text.split())
        self.out.append(f"**{text}**" if bold_lead else text)

    def h1(self, text: str) -> None:
        self.out.append(f"## {text}")

    def h2(self, _at, text: str) -> None:
        self.out.append(f"### {text}")

    def code(self, _at, text: str) -> None:
        self.out.append("```\n" + text.strip("\n") + "\n```")

    def table(self, _at, header: list[str], rows: list[list]) -> None:
        # One block, not one entry per row: render() joins entries with a blank
        # line, and a blank line between rows stops Markdown seeing a table.
        lines = ["| " + " | ".join(str(h) for h in header) + " |",
                 "|" + "|".join("---" for _ in header) + "|"]
        lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        self.out.append("\n".join(lines))

    def caption(self, _at, text: str) -> None:
        self.table_n += 1
        self.out.append(f"*Table {self.table_n}: {text}*")

    def replace(self, pred, text: str) -> None:
        self.edits.append((self._describe(pred), text))

    @staticmethod
    def _describe(pred) -> str:
        """Recover the literal the lambda tests for, so the edit list is actionable."""
        try:
            import inspect
            src = inspect.getsource(pred)
            m = (re.search(r'startswith\(\s*"([^"]+)"', src)
                 or re.search(r'==\s*"([^"]+)"', src))
            return m.group(1) if m else "<paragraph>"
        except Exception:
            return "<paragraph>"

    # Same operations, named for readability at the end of the document.
    def h2_end(self, text): self.h2(None, text)
    def para_end(self, text): self.para(None, text)
    def code_end(self, text): self.code(None, text)
    def table_end(self, header, rows): self.table(None, header, rows)
    def caption_end(self, text): self.caption(None, text)

    @staticmethod
    def pct(x) -> str:
        return f"{float(x) * 100:.0f}%"

    def render(self) -> str:
        body = "\n\n".join(self.out)
        edits = "\n\n".join(
            f"**Replace the paragraph beginning** “{anchor}…”\n\n> {text}"
            for anchor, text in self.edits)
        return HEADER + body + "\n\n---\n\n" + EDITS_HEADER + edits + "\n"


HEADER = """<!-- Generated by scripts/build_report.py from committed evidence.
     Do not hand-edit; rerun the script instead. -->

# Part B report — Stages 2 to 4

These sections extend `docs/PartB_Draft_STT.docx`, which covers Stage 1. Paste
them into that document under the headings shown, applying Word's Heading 2
style to the `###` headings and the document's existing table formatting to the
tables.

**Fix the table numbering in the Stage 1 draft first.** Its captions currently
run 1, 2, 3, **5**, 6, **4**, **5** — Table 5 appears twice and Table 4 is out of
order. Renumber them 1–6 in document order; the tables below continue from 7.

"""

EDITS_HEADER = """## Paragraphs to replace in the Stage 1 draft

The draft states in several places that it covers Stage 1 only, and its
Appendix A is an unfilled placeholder. Replace each paragraph below.

"""


# -------------------------------------------------------------------------- run

def build() -> str:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import report_backmatter
    import report_sections

    e = evidence()
    D = Markdown()
    report_sections.front_matter(D, e)
    report_sections.deviations(D, e)
    D.h1("Pseudocode (continued)")
    report_sections.pseudocode(D, e)
    D.h1("Algorithm Analysis (continued)")
    report_sections.analysis(D, e)
    D.h1("Coding (continued)")
    report_backmatter.coding(D, e)
    D.h1("References to add")
    report_backmatter.references(D, e)
    D.h1("Appendices")
    report_backmatter.appendices(D, e)
    return D.render()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="rebuild and report whether the committed file is current")
    args = ap.parse_args()

    text = build()
    if args.check:
        if not OUT.exists():
            print(f"{OUT.relative_to(ROOT)} does not exist")
            return 1
        if OUT.read_text(encoding="utf-8") == text:
            print(f"{OUT.relative_to(ROOT)} is up to date with the evidence")
            return 0
        print(f"{OUT.relative_to(ROOT)} is STALE - rerun without --check")
        return 1

    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(text.split()):,} words, "
          f"{text.count(chr(10) + '### ')} subsections, "
          f"{text.count(chr(10) + '*Table ')} tables)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
