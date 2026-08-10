"""Check that app/stt.py reproduces the evaluated pipeline.

`app/stt.py --check-config` compares constants. This compares behaviour: it runs the
application's own transcribe() over clips from the evaluation set and diffs the output
against the transcripts stored in results_detail.csv for malaysian_prompt_routed.

That matters because the reported 0.349 WER is a claim about the evaluated pipeline.
If the application produces different text from the same audio, the claim does not
transfer to the demo, and the two could diverge for reasons no constant check would
catch -- a different generation flag, a lost prompt, a changed strip.

Exact string equality is the target but not the pass condition: the stored transcripts
were produced on Colab's GPU in float32, and reproducing them on a laptop CPU can move
a token. The script reports exact matches, near matches, and real divergences
separately so you can judge which you are looking at.

Usage
    python scripts/verify_stt_matches_eval.py                # 5 clips, mixed categories
    python scripts/verify_stt_matches_eval.py --limit 48     # the whole set (slow on CPU)
    python scripts/verify_stt_matches_eval.py --category en_dom
"""

import argparse
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MANIFEST = REPO / "eval" / "speech" / "manifest.csv"
CONFIG = "malaysian_prompt_routed"


def find_results() -> Path:
    for p in (REPO / "eval" / "speech" / "results_detail.csv",
              REPO / "eval" / "speech" / "results" / "results_detail.csv"):
        if p.exists():
            return p
    sys.exit("results_detail.csv not found")


def find_audio(rel_path: str, speaker: str) -> Path | None:
    """Audio is flat in Colab and split by speaker on the laptop. Accept both."""
    name = Path(rel_path).name
    for cand in (REPO / "eval" / "speech" / rel_path,
                 REPO / "eval" / "speech" / "audio" / name,
                 REPO / "eval" / "speech" / "audio" / speaker / name):
        if cand.exists():
            return cand
    return None


def norm(s: str) -> str:
    return " ".join(str(s).lower().replace(",", " ").replace(".", " ")
                    .replace("?", " ").replace("!", " ").split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=6,
                    help="default 6 = one clip per switch type")
    ap.add_argument("--category", help="restrict to one switch type")
    args = ap.parse_args()

    from app import stt

    man = pd.read_csv(MANIFEST)
    man = man[man["phase"].astype(str) == "1"]
    stored = pd.read_csv(find_results())
    stored = stored[stored.config == CONFIG].set_index("id")

    if args.category:
        man = man[man.switch_type == args.category]
    else:
        # one clip per category first, so a small --limit still covers the spread
        man = man.groupby("switch_type", group_keys=False).head(
            max(1, args.limit // man.switch_type.nunique() + 1))
    man = man.head(args.limit)

    print(f"comparing {len(man)} clip(s) against stored {CONFIG} transcripts\n")
    stt.warm_up()

    exact = near = diverged = missing = 0
    lang_mismatch = 0

    for r in man.itertuples():
        audio = find_audio(r.audio_path, r.speaker)
        if audio is None:
            print(f"  --  {r.id:8} audio not found ({r.audio_path})")
            missing += 1
            continue
        if r.id not in stored.index:
            print(f"  --  {r.id:8} no stored transcript")
            missing += 1
            continue

        got = stt.transcribe_detailed(audio)
        want_text = str(stored.loc[r.id, "hypothesis"])
        want_lang = str(stored.loc[r.id, "detected_lang"])

        ratio = SequenceMatcher(None, norm(got.text), norm(want_text)).ratio()
        if norm(got.text) == norm(want_text):
            tag, exact = "OK  exact", exact + 1
        elif ratio >= 0.9:
            tag, near = "~   near ", near + 1
        else:
            tag, diverged = "BAD diverge", diverged + 1

        if got.language != want_lang:
            lang_mismatch += 1

        print(f"  {tag} {r.id:8} [{r.switch_type:9}] lang {got.language}"
              f"{'' if got.language == want_lang else f' != stored {want_lang}'}"
              f"  p_en={got.p_en:.3f}  sim={ratio:.2f}  {got.seconds:.1f}s")
        if tag.startswith("BAD") or tag.startswith("~"):
            print(f"        app   : {got.text}")
            print(f"        stored: {want_text}")

    n = exact + near + diverged
    print(f"\n{exact} exact, {near} near (>=0.90), {diverged} diverged, {missing} skipped")
    if lang_mismatch:
        print(f"{lang_mismatch} routing decision(s) differ from the stored run — investigate,"
              " the detector should be deterministic")

    if n and diverged == 0 and lang_mismatch == 0:
        print("\nPASS — app/stt.py reproduces the evaluated pipeline.")
        return 0
    if n == 0:
        print("\nNothing compared.")
        return 1
    print("\nFAIL — the application and the evaluation harness disagree.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
