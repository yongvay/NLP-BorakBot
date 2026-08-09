"""Convert and rename phone recordings into the filenames the eval script expects.

Phones name files 'Recording 007.mp4'; the manifest expects 's07_ms_dom_yv.wav'.
Renaming 24 of those by hand is where a misalignment creeps in -- and a misaligned
clip is worse than a missing one, because it scores as a catastrophic transcription
error rather than showing up as an obvious gap.

This pairs raw files to manifest rows IN ORDER, so it only works if you recorded the
sentences in the order they appear in the spreadsheet. It refuses to run if the counts
do not match, and prints the pairing for you to check before touching anything.

Usage:
    python scripts/prepare_audio.py --speaker yv --raw "C:/path/to/raw"   # preview
    python scripts/prepare_audio.py --speaker yv --raw "C:/path/to/raw" --go
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

AUDIO_EXT = {".mp4", ".m4a", ".mp3", ".wav", ".aac", ".3gp", ".ogg", ".opus"}
REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "eval" / "speech" / "manifest.csv"
OUT_DIR = REPO / "eval" / "speech" / "audio"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True, choices=["yv", "pq"])
    ap.add_argument("--raw", required=True, help="folder holding the phone recordings")
    ap.add_argument("--phase", default="1")
    ap.add_argument("--by-name", action="store_true",
                    help="pair by filename order instead of recording time")
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
    # Recording order is the ground truth for pairing, so sort by modification time.
    # --by-name is the fallback for phones that rewrite timestamps on transfer.
    raw.sort(key=(lambda p: p.name.lower()) if args.by_name else (lambda p: p.stat().st_mtime))

    print(f"manifest rows : {len(rows)}   (speaker={args.speaker}, phase={args.phase})")
    print(f"raw files     : {len(raw)}   in {raw_dir}")
    print(f"paired by     : {'filename' if args.by_name else 'recording time'}\n")

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
        print("\nPREVIEW ONLY. Check every line above -- especially the last one, since an\n"
              "off-by-one shifts every single pairing. Re-run with --go when it looks right.")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src, r in zip(raw, rows):
        dst = REPO / "eval" / "speech" / r["audio_path"]
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
