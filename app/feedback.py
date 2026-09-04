"""Human-in-the-loop corrections — Stage 4 of the pipeline.

    reply  ->  thumbs up/down  ->  feedback.db  ->  reviewed in batches  ->  next round

Part A objective 6 and pipeline stage 4. Every reply carries a verdict; on
thumbs-down the user supplies the answer they expected, and the whole exchange
is logged with the context that produced it.

WHAT THIS DELIBERATELY DOES NOT DO

**Corrections do not enter the training set automatically.** They are logged
immediately and appended only in periodic, team-reviewed batches. Part A §5.7
took that position from the literature on human-in-the-loop correction: an
unmoderated loop trains on whatever it is told, including careless and
adversarial input, and a chatbot that can be taught a wrong fact by one user is
a worse system than one that cannot learn at all. The moderation step is the
design, not a shortcut around building one.

So there is no retraining trigger here and no auto-export into
`stage2_chatbot/corpus/`. `--export` prints the corrections; a human decides what
becomes a training pair.

WHY THE SCHEMA SPLITS transcript FROM user_text

`transcript` is what Whisper heard, NULL when the question was typed.
`user_text` is what the model was given, after `app/normalise.py`. A wrong
answer to a misheard question is an ASR failure and belongs in the WER column,
not the NLP column — DESIGN.md §5 reports the two apart, and one merged field
would make that impossible after the fact.

Storage is `sqlite3` from the standard library: no new dependency, one file,
and the file is gitignored while `stage2_chatbot/schema.sql` is committed.

Usage
    python app/feedback.py --init          # create an empty feedback.db
    python app/feedback.py --recent        # last 20 entries
    python app/feedback.py --export        # corrections, as review-ready JSON
    python app/feedback.py --self-test     # round-trip check, uses a temp file

Deleting is deliberately not here. It is a destructive action on a gitignored
database, and the only person who should be doing it is the one already
looking at the rows -- so it lives in app/pages/1_Admin.py, on the table
itself, and nowhere else.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "stage2_chatbot" / "schema.sql"
DB = ROOT / "feedback.db"


@dataclass
class Entry:
    id: int
    created_utc: str
    transcript: str | None
    user_text: str
    reply: str
    verdict: str
    correction: str | None
    context: list[dict]
    backend: str | None
    seconds: float | None


@contextmanager
def _db(db: Path = DB):
    """Commit on success, roll back on error, and always close.

    `with sqlite3.connect(...) as conn` commits but does NOT close -- a leak that
    is invisible on Linux and immediate on Windows, where the open handle blocks
    the file. Streamlit re-runs this module's callers on every interaction, so a
    leak per click is not hypothetical.
    """
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init(db: Path = DB) -> None:
    """Create the table if it is absent. Safe to call on every app start."""
    with _db(db) as conn:
        conn.executescript(SCHEMA.read_text(encoding="utf-8"))


def log(
    user_text: str,
    reply: str,
    verdict: str,
    *,
    transcript: str | None = None,
    correction: str | None = None,
    context: list[dict] | None = None,
    backend: str | None = None,
    seconds: float | None = None,
    db: Path = DB,
) -> int:
    """Record one verdict. Returns the row id."""
    if verdict not in ("up", "down"):
        raise ValueError(f"verdict must be 'up' or 'down', got {verdict!r}")

    with _db(db) as conn:
        cur = conn.execute(
            """INSERT INTO feedback
               (created_utc, transcript, user_text, reply, verdict,
                correction, context, backend, seconds)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                transcript,
                user_text,
                reply,
                verdict,
                correction or None,
                json.dumps(context or [], ensure_ascii=False),
                backend,
                seconds,
            ),
        )
        return int(cur.lastrowid)


def _row(r: sqlite3.Row) -> Entry:
    return Entry(
        id=r["id"],
        created_utc=r["created_utc"],
        transcript=r["transcript"],
        user_text=r["user_text"],
        reply=r["reply"],
        verdict=r["verdict"],
        correction=r["correction"],
        context=json.loads(r["context"] or "[]"),
        backend=r["backend"],
        seconds=r["seconds"],
    )


def recent(limit: int = 20, db: Path = DB) -> list[Entry]:
    if not db.exists():
        return []
    with _db(db) as conn:
        rows = conn.execute(
            "SELECT * FROM feedback ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row(r) for r in rows]


def counts(db: Path = DB) -> dict[str, int]:
    """{'up': n, 'down': n}. For the sidebar, so the loop is visible."""
    if not db.exists():
        return {"up": 0, "down": 0}
    with _db(db) as conn:
        rows = conn.execute(
            "SELECT verdict, COUNT(*) AS n FROM feedback GROUP BY verdict"
        ).fetchall()
    out = {"up": 0, "down": 0}
    out.update({r["verdict"]: r["n"] for r in rows})
    return out


def export_corrections(db: Path = DB) -> list[dict]:
    """Thumbs-down rows that carry a correction, shaped like a training pair.

    Shaped like one, not saved as one -- see the module docstring. A person
    reviews these before any of them reaches stage2_chatbot/corpus/.
    """
    if not db.exists():
        return []
    with _db(db) as conn:
        rows = conn.execute(
            """SELECT * FROM feedback
               WHERE verdict = 'down' AND correction IS NOT NULL
               ORDER BY id""",
        ).fetchall()
    return [
        {
            "user": r["user_text"],
            "assistant": r["correction"],
            "_source": "feedback",
            "_id": r["id"],
            "_logged_utc": r["created_utc"],
            "_rejected_reply": r["reply"],
            "_transcript": r["transcript"],
        }
        for r in rows
    ]


def delete(ids: list[int], db: Path = DB) -> int:
    """Remove rows by id. Returns how many were actually deleted.

    A hard delete, not a soft `deleted` flag. The log is a record of testing,
    not an audit trail anyone is accountable to, and a flag would mean every
    reader -- recent(), counts(), export_corrections(), the admin table --
    grows a WHERE clause to stay correct. One line of SQL against five callers
    that each have to remember it is the wrong trade here.

    Ids are AUTOINCREMENT, so deleting row 2 leaves rows 1 and 3 as they are and
    the next insert still gets a fresh id. Nothing is renumbered and no existing
    id changes meaning -- an exported correction stays traceable to its `_id`.

    The only caller is the review page. Confirmation is that page's job, not
    this function's.
    """
    if not ids:
        return 0
    with _db(db) as conn:
        # The f-string interpolates only the '?,?,?' placeholder run, never the
        # ids themselves -- those still go through the parameter binding.
        cur = conn.execute(
            f"DELETE FROM feedback WHERE id IN ({','.join('?' * len(ids))})",
            ids,
        )
        return cur.rowcount


# ------------------------------------------------------------------- self-test

def self_test() -> int:
    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "test_feedback.db"
    init(tmp)

    a = log("nak renew roadtax", "Boleh renew kat MyEG.", "up",
            backend="cpu-bf16", seconds=70.2, db=tmp)
    b = log("berapa lama pakai P", "Maaf, saya tak pasti.", "down",
            transcript="berapa lama pakai P?",
            correction="Tempoh PDL 2 years.",
            context=[{"role": "user", "content": "hi"}],
            backend="cpu-bf16", seconds=68.9, db=tmp)

    checks = []
    rows = recent(db=tmp)
    checks.append(("two rows logged", len(rows) == 2))
    checks.append(("ids returned", a == 1 and b == 2))
    checks.append(("newest first", rows[0].id == 2))
    checks.append(("context round-trips", rows[0].context == [{"role": "user", "content": "hi"}]))
    checks.append(("transcript kept apart", rows[0].transcript == "berapa lama pakai P?"))
    checks.append(("counts", counts(db=tmp) == {"up": 1, "down": 1}))

    exported = export_corrections(db=tmp)
    checks.append(("one correction exported", len(exported) == 1))
    checks.append(("shaped as a training pair",
                   exported[0]["user"] == "berapa lama pakai P"
                   and exported[0]["assistant"] == "Tempoh PDL 2 years."))

    bad = False
    try:
        log("x", "y", "maybe", db=tmp)
    except ValueError:
        bad = True
    checks.append(("bad verdict rejected", bad))

    init(tmp)  # idempotent
    checks.append(("init is idempotent", len(recent(db=tmp)) == 2))

    checks.append(("delete ignores absent ids", delete([99], db=tmp) == 0))
    checks.append(("delete removes one row", delete([2], db=tmp) == 1))
    checks.append(("survivor untouched", [e.id for e in recent(db=tmp)] == [1]))
    checks.append(("correction went with it", export_corrections(db=tmp) == []))
    checks.append(("counts follow", counts(db=tmp) == {"up": 1, "down": 0}))
    c3 = log("after delete", "reply", "up", db=tmp)
    checks.append(("ids are not reused", c3 == 3))

    failed = 0
    for name, ok in checks:
        failed += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}")
    print(f"\n{len(checks) - failed}/{len(checks)} passed")

    tmp.unlink(missing_ok=True)
    return 1 if failed else 0


# -------------------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect the BorakBot feedback log.")
    ap.add_argument("--init", action="store_true", help="create an empty database")
    ap.add_argument("--recent", action="store_true", help="print the last 20 entries")
    ap.add_argument("--export", action="store_true",
                    help="print corrections as review-ready JSON")
    ap.add_argument("--out", type=Path, default=None, metavar="PATH",
                    help="write --export to a UTF-8 file instead of stdout. Use this "
                         "rather than a shell redirect: PowerShell's '>' writes a BOM, "
                         "and json.load chokes on it.")
    ap.add_argument("--self-test", action="store_true", help="round-trip check")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.init:
        init()
        print(f"ready: {DB.relative_to(ROOT)}")
        return 0

    if args.export:
        rows = export_corrections()
        blob = json.dumps(rows, ensure_ascii=False, indent=2)
        if args.out:
            args.out.write_text(blob + "\n", encoding="utf-8")
            print(f"wrote {args.out}  ({len(rows)} correction(s) awaiting review)")
        else:
            print(blob)
            print(f"\n{len(rows)} correction(s) awaiting review", file=sys.stderr)
        return 0

    if args.recent:
        rows = recent()
        if not rows:
            print("no feedback logged yet")
            return 0
        c = counts()
        print(f"{c['up']} up, {c['down']} down\n")
        for e in rows:
            mark = "+" if e.verdict == "up" else "-"
            print(f"{mark} [{e.id}] {e.created_utc}  ({e.backend}, {e.seconds:.0f}s)"
                  if e.seconds else f"{mark} [{e.id}] {e.created_utc}")
            print(f"    Q: {e.user_text}")
            print(f"    A: {e.reply[:100]}")
            if e.correction:
                print(f"    -> should be: {e.correction}")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())