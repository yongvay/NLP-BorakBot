"""Convert and rename phone recordings into the filenames the eval script expects.

Phones name files 'Recording 007.mp4'; the manifest expects 's07_ms_dom_yv.wav'.
Renaming 24 of those by hand is where a misalignment creeps in -- and a misaligned
clip is worse than a missing one, because it scores as a catastrophic transcription
error rather than showing up as an obvious gap.

This pairs raw files to manifest rows IN ORDER, so it only works if you recorded the
sentences in the order they appear in the spreadsheet. It refuses to run if the counts
do not match, and prints the pairing for you to check before touching anything.

Usage:
    python archive/scripts/prepare_audio.py --speaker yv --raw "C:/path/to/raw"   # preview
    python archive/scripts/prepare_audio.py --speaker yv --raw "C:/path/to/raw" --go
"""

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path


def natural_key(p: Path):
    """Sort '2.m4a' before '10.m4a'.

    Plain alphabetical sort compares character by character, so '10' < '2' because
    '1' < '2'. Splitting on digit runs and comparing the numbers as integers gives
    the ordering a human expects. Works for '1.m4a' and 'Recording 001.m4a' alike.
    """
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name.lower())]

AUDIO_EXT = {".mp4", ".m4a", ".mp3", ".wav", ".aac", ".3gp", ".ogg", ".opus"}
SPEECH = Path(__file__).resolve().parent.parent / "eval" / "speech"   # archive/eval/speech
MANIFEST = SPEECH / "manifest.csv"
OUT_DIR = SPEECH / "audio"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True, choices=["yv", "pq"])
    ap.add_argument("--raw", required=True, help="folder holding the phone recordings")
    ap.add_argument("--phase", default="1")
    ap.add_argument("--by-name", action="store_true",
                    help="pair by filename order (numeric-aware) instead of recording time")
    ap.add_argument("--force-time", action="store_true",
                    help="convert using timestamp order even when it disagrees with filenames")
    ap.add_argument("--go", action="store_true", help="actually convert (default: preview)")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found. Install it:  winget install Gyan.FFmpeg")
        print("then close and reopen your terminal.")
        return 1

    rows = [r for r in csv.DictReader(MANIFEST.open(encoding="utf-8"))
            if r["speaker"] == args.speaker and r["phase"] == args.phase]

    raw_dir = Path(args.raw)
    if not raw_dir.is_dir():
        print(f"not a folder: {raw_dir}")
        return 1

    raw = [p for p in raw_dir.iterdir() if p.suffix.lower() in AUDIO_EXT]
    by_time = sorted(raw, key=lambda p: p.stat().st_mtime)
    by_name = sorted(raw, key=natural_key)
    raw = by_name if args.by_name else by_time

    print(f"manifest rows : {len(rows)}   (speaker={args.speaker}, phase={args.phase})")
    print(f"raw files     : {len(raw)}   in {raw_dir}")
    print(f"paired by     : {'filename' if args.by_name else 'recording time'}\n")

    # Copying off a phone often rewrites modification times into arbitrary order, which
    # silently destroys time-based pairing. If the filenames are numbered and disagree
    # with the timestamps, the timestamps are the suspect one -- say so loudly, because
    # the output/reference columns always agree with each other and hide the problem.
    disagree = [p.name for p in by_time] != [p.name for p in by_name]
    if not args.by_name and disagree:
        print("!" * 78)
        print("WARNING: timestamp order does NOT match filename order.")
        print("Copying from a phone commonly rewrites modification times. If your files")
        print("are numbered in recording order, re-run with --by-name instead.")
        print("!" * 78 + "\n")
        if args.go and not args.force_time:
            # Refuse rather than warn. A warning printed above 24 lines of output is easy
            # to scroll past, and the result -- 24 correctly named files containing the
            # wrong audio -- looks like success. Silent corruption beats a loud failure
            # only if you never have to trust the data afterwards.
            print("REFUSING TO CONVERT.")
            print("Pass --by-name as well if the filenames are in recording order:")
            print(f'  python archive/scripts/prepare_audio.py --speaker {args.speaker} '
                  f'--raw "{args.raw}" --by-name --go')
            print("Or pass --force-time if you really do want timestamp order.")
            return 1

    if len(raw) != len(rows):
        print("COUNT MISMATCH -- refusing to guess.")
        print("Re-record the missing ones, or delete stray files, then run again.")
        print("If you recorded out of order, rename the raw files and use --by-name.")
        return 1

    print(f"{'raw file':<34} -> {'output':<30} reference")
    print("-" * 110)
    for src, r in zip(raw, rows):
        print(f"{src.name:<34} -> {Path(r['audio_path']).name:<30} {r['reference'][:40]}")

    if not args.go:
        print("\nPREVIEW ONLY.")
        print("Check the LEFT column -- the raw filenames. 'output' and 'reference' both come")
        print("from the manifest, so they always agree with each other and prove nothing.")
        print("The raw files must run in your recording order, top to bottom.")
        print("Re-run with --go when that column looks right.")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src, r in zip(raw, rows):
        dst = SPEECH / r["audio_path"]
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-vn",                  # drop video stream if the mp4 has one
             "-ar", "16000",         # Whisper resamples to 16 kHz anyway
             "-ac", "1",             # mono
             "-c:a", "pcm_s16le",    # standard 16-bit PCM wav
             str(dst)],
            check=True)
        print(f"ok  {dst.name}")

    print(f"\ndone: {len(rows)} files in {OUT_DIR}")
    print("Now tick 'recorded' in recording_tracker.xlsx.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
