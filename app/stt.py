"""Speech-to-text for BorakBot — Stage 1 of the pipeline, in deployment form.

`eval/speech/run_wer.py` is an evaluation harness: it reads a manifest, sweeps nine
configurations and scores WER. The application needs one configuration and one
function. This module is that extraction.

The shipped configuration is `malaysian_prompt_routed`, the best of the nine
(0.349 strict WER, 0.339 lenient, 48 clips / 387 reference words):

    vanilla whisper-small    ->  language identification, one encoder pass
    mesolitica/...-v3        ->  transcription, using the chosen language token
    ROJAK_PROMPT             ->  decoder context, supplied to the transcriber

WHY THE SETTINGS ARE DUPLICATED RATHER THAN IMPORTED
----------------------------------------------------
The application must not depend on the evaluation harness -- `run_wer.py` pulls in
pandas and jiwer, which a Streamlit deployment has no reason to install. But a copy
that silently drifts from the evaluated settings is worse than a dependency: the
reported 0.349 would stop describing what the demo actually runs, and that is a fair
question at the individual assessment.

So the constants are duplicated *and checked*. `python app/stt.py --check-config`
parses run_wer.py and asserts every shared setting matches. Run it after touching
either file.

Usage
    python app/stt.py recording.wav              # transcribe a file
    python app/stt.py recording.wav --json       # with language and timing
    python app/stt.py --check-config             # verify against the eval harness
    python app/stt.py --warm-up                  # download and cache the models

    from app.stt import transcribe
    text = transcribe("recording.wav")
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Optional, Union

# --------------------------------------------------------------------- settings
# Mirrored from eval/speech/run_wer.py. Changing either without the other makes the
# reported WER inapplicable to the running system -- see --check-config.

VANILLA_MODEL = "small"
MALAYSIAN_MODEL = "mesolitica/malaysian-whisper-small-v3"

# English is only selected when the detector is confident. The two failure directions
# are not symmetric: a wrong "en" on Malay audio is usually harmless but occasionally
# unbounded (one evaluation clip produced 362 insertions against a seven-word
# reference), whereas a wrong "ms" on English audio merely translates it (0.788 WER).
# Chosen a priori, not tuned on the test set. See report §5.
EN_THRESHOLD = 0.5

# Supplied to the decoder as context. It must itself be code-switched: an earlier,
# predominantly Malay prompt pushed English-dominant utterances into translation.
# Shares no vocabulary with the evaluation set.
ROJAK_PROMPT = (
    "cuaca panas gila hari ni kan, tak larat nak keluar. "
    "can you send me the file later, i need to check something first. "
    "eh awak dah reply that email ke belum lah. "
    "so how ah, do you want to join us or not."
)

AudioInput = Union[str, Path, bytes, IO[bytes]]

_models: dict = {}


# ------------------------------------------------------------------- model cache
# Neither checkpoint ships with the repo (.gitignore forbids weights), so the first
# run downloads about 930 MB and every run after that reads it back from the paths
# below. Distinguishing those two cases is the whole point of this section: a fresh
# process spends ~13 s loading the weights into RAM, which looks exactly like a
# download unless the interface says otherwise. See streamlit_app.py.

def _vanilla_checkpoint() -> Path:
    """Where openai-whisper keeps small.pt.

    Mirrors the resolution inside `whisper.load_model` -- XDG_CACHE_HOME if set,
    otherwise ~/.cache -- so the file probed here is the file it would load.
    """
    root = Path(os.getenv("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    return root / "whisper" / f"{VANILLA_MODEL}.pt"


def _malaysian_snapshot() -> Optional[Path]:
    """The cached Mesolitica snapshot directory, or None if never downloaded.

    Asks huggingface_hub rather than globbing the cache: the hub layout splits
    blobs from snapshots and reaches them through a `refs/main` indirection, and
    reimplementing that here would break the first time the layout changed.
    """
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(MALAYSIAN_MODEL, "config.json")
    return Path(hit).parent if isinstance(hit, str) else None   # else: miss sentinel


def _dir_size(d: Path) -> int:
    """Bytes on disk under `d`. stat() follows the hub's blob links, so this is real."""
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())


def human_bytes(n: int) -> str:
    return f"{n / 1e6:.0f} MB" if n < 1e9 else f"{n / 1e9:.2f} GB"


def cache_status() -> dict:
    """Whether each model is already on disk, where, and how big.

    Returned rather than printed so the CLI and the Streamlit sidebar can report
    the same facts. `cached` false means the next load hits the network.
    """
    ckpt = _vanilla_checkpoint()
    snap = _malaysian_snapshot()
    return {
        "vanilla": {
            "name": VANILLA_MODEL,
            "cached": ckpt.is_file(),
            "path": ckpt,
            "bytes": ckpt.stat().st_size if ckpt.is_file() else 0,
        },
        "malaysian": {
            "name": MALAYSIAN_MODEL,
            "cached": snap is not None,
            "path": snap,
            "bytes": _dir_size(snap) if snap is not None else 0,
        },
    }


# ------------------------------------------------------------------ model access

def _vanilla():
    """Base Whisper. Used only as a language detector, never to transcribe."""
    if "vanilla" not in _models:
        import whisper

        # Load by PATH when the checkpoint is already cached, by NAME only to fetch
        # it the first time. The name branch routes through whisper's _download(),
        # which reads all 483 MB into memory and recomputes its SHA256 on every
        # process start -- before torch.load reopens the same file and reads it
        # again. The path branch skips that (1.90 s -> 1.14 s here). What it gives
        # up is set_alignment_heads, which exists only to produce word-level
        # timestamps; this module calls detect_language and nothing else.
        ckpt = _vanilla_checkpoint()
        _models["vanilla"] = whisper.load_model(
            str(ckpt) if ckpt.is_file() else VANILLA_MODEL
        )
    return _models["vanilla"]


@contextmanager
def _offline(enabled: bool):
    """Stop huggingface_hub reaching for the network inside this block.

    The obvious `local_files_only=True` cannot be used here: `pipeline()` forwards
    kwargs it does not recognise all the way down to `generate()`, which rejects it
    at transcription time -- so the model loads and the first clip crashes.

    Setting HF_HUB_OFFLINE in the environment is not enough either. huggingface_hub
    evaluates it into `constants.HF_HUB_OFFLINE` at import, and cache_status() has
    already imported the package by the time we get here. What both libraries read
    per request is `constants.is_offline_mode()`, which returns that module
    attribute -- so patching the attribute is what actually takes effect. The
    environment variable is set alongside it only to keep the two consistent.
    """
    if not enabled:
        yield
        return

    from huggingface_hub import constants

    was, env = constants.HF_HUB_OFFLINE, os.environ.get("HF_HUB_OFFLINE")
    constants.HF_HUB_OFFLINE = True
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        constants.HF_HUB_OFFLINE = was
        if env is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = env


def _malaysian():
    """The Malaysian fine-tune. Does all the actual transcription."""
    if "malaysian" not in _models:
        import torch
        from transformers import pipeline

        def build():
            return pipeline(
                "automatic-speech-recognition",
                model=MALAYSIAN_MODEL,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float32,
            )

        # Read straight from the cache once it is populated. Left to itself the
        # pipeline makes an unauthenticated revision check against the Hub on every
        # start: wasted time on a good network, and a stall on a bad one. Campus
        # wi-fi during the demo is the wrong place to discover that.
        cached = cache_status()["malaysian"]["cached"]
        try:
            with _offline(cached):
                _models["malaysian"] = build()
        except OSError:
            # Every offline miss lands here -- LocalEntryNotFoundError and
            # OfflineModeIsEnabled are both OSError subclasses. Reached when the
            # cache is present but incomplete, as an interrupted first download
            # leaves config.json behind without the weights. Retry with the network.
            if not cached:
                raise
            _models["malaysian"] = build()
    return _models["malaysian"]


def warm_up() -> None:
    """Load both models. Call once at application start.

    Without this the first user to press record waits through two model downloads
    and thinks the app has hung.
    """
    _vanilla()
    _malaysian()


# ------------------------------------------------------------------------- input

def _as_path(audio: AudioInput) -> tuple[str, tempfile.NamedTemporaryFile | None]:
    """Normalise any accepted input to a filesystem path.

    Whisper's loader shells out to ffmpeg and needs a real file, but Streamlit's
    recorder hands back an in-memory buffer. The temporary file is returned so the
    caller can delete it; letting it fall out of scope would delete it too early on
    Windows, where an open handle blocks the read.
    """
    if isinstance(audio, (str, Path)):
        p = Path(audio)
        if not p.exists():
            raise FileNotFoundError(f"no such audio file: {p}")
        return str(p), None

    data = audio if isinstance(audio, bytes) else audio.read()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name, tmp


# -------------------------------------------------------------------- the stages

def detect_language(audio: AudioInput) -> tuple[str, float]:
    """Choose the language token for an utterance. Returns (choice, p_en).

    The Mesolitica checkpoint transcribes English competently when told to, but its
    own detector always answers Malay, so English clips come back translated. Base
    Whisper's detector is unbiased and costs one encoder pass.

    The probability is returned alongside the decision because the decision alone
    cannot distinguish a clip missed narrowly from one missed completely.
    """
    import whisper

    path, tmp = _as_path(audio)
    try:
        m = _vanilla()
        mel = whisper.log_mel_spectrogram(
            whisper.pad_or_trim(whisper.load_audio(path)),
            n_mels=m.dims.n_mels,
        ).to(m.device)
        _, probs = m.detect_language(mel)
        top = max(probs, key=probs.get)
        p_en = float(probs.get("en", 0.0))
        return ("en" if top == "en" and p_en > EN_THRESHOLD else "ms"), p_en
    finally:
        if tmp is not None:
            Path(tmp.name).unlink(missing_ok=True)


def _strip_prompt(text: str, prompt: str) -> str:
    """Remove prompt text echoed into the transcript.

    Transformers echoes `prompt_ids` back into the output, sometimes whole, sometimes
    truncated, and not always at position 0. Left in, every echoed word counts as an
    insertion -- which is how one evaluation configuration reached 106% WER with 215
    insertions on a 387-word corpus. Checking only `startswith` is not enough.

    Currently inert: none of the 288 prompted transcripts in the evaluation results
    takes either branch, so whatever caused the echoing has since stopped. It is kept
    because the behaviour is transformers-version-dependent and its silent return
    would be expensive.
    """
    t = " ".join(text.split())
    p = " ".join(prompt.split())
    low_t, low_p = t.lower(), p.lower()

    if low_p in low_t:                      # whole prompt echoed somewhere
        i = low_t.index(low_p)
        return " ".join((t[:i] + " " + t[i + len(p):]).split())

    # Truncated echo at the head. Cut at the exact end of the shared prefix rather
    # than probing a coarse grid: stepping in fives stops up to four characters short
    # and leaves a fragment of the prompt's last word at the front of the transcript.
    n = 0
    while n < min(len(low_t), len(low_p)) and low_t[n] == low_p[n]:
        n += 1
    if n >= 16:                             # guard against short coincidental overlap
        return t[n:].lstrip(" ,.-").strip()
    return t


@dataclass
class Transcription:
    text: str
    language: str          # the token handed to the transcriber: "en" or "ms"
    p_en: float            # detector confidence, for logging and debugging
    seconds: float         # wall-clock, for the demo's latency discussion


def transcribe_detailed(audio: AudioInput) -> Transcription:
    """Full pipeline, returning the routing decision alongside the text."""
    started = time.perf_counter()
    path, tmp = _as_path(audio)
    try:
        lang, p_en = detect_language(path)

        asr = _malaysian()
        kwargs = {"task": "transcribe", "language": lang}

        # Mesolitica pins the Malay token in generation_config, which is why omitting
        # `language=` produced byte-identical output to forcing Malay. Passing it
        # explicitly is what makes the routed choice take effect.
        prompt_ids = asr.tokenizer.get_prompt_ids(ROJAK_PROMPT, return_tensors="pt")
        kwargs["prompt_ids"] = prompt_ids.to(asr.model.device)

        raw = asr(path, generate_kwargs=kwargs)["text"].strip()
        text = _strip_prompt(raw, ROJAK_PROMPT)

        return Transcription(text=text, language=lang, p_en=p_en,
                             seconds=round(time.perf_counter() - started, 2))
    finally:
        if tmp is not None:
            Path(tmp.name).unlink(missing_ok=True)


def transcribe(audio: AudioInput) -> str:
    """Audio in, rojak text out. The function the rest of the app calls."""
    return transcribe_detailed(audio).text


# ------------------------------------------------------- configuration guard-rail

def check_config() -> int:
    """Assert every shared setting matches eval/speech/run_wer.py.

    Parsed rather than imported, so this works without pandas or jiwer installed.
    """
    import ast

    harness = Path(__file__).resolve().parent.parent / "eval" / "speech" / "run_wer.py"
    if not harness.exists():
        print(f"cannot find {harness}")
        return 1

    tree = ast.parse(harness.read_text(encoding="utf-8"))
    found: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass

    checks = [
        ("ROJAK_PROMPT", ROJAK_PROMPT, found.get("ROJAK_PROMPT")),
        ("EN_THRESHOLD", EN_THRESHOLD, found.get("EN_THRESHOLD")),
        ("vanilla model", VANILLA_MODEL, found.get("VANILLA")),
        ("malaysian model", MALAYSIAN_MODEL, found.get("MALAYSIAN")),
    ]

    bad = []
    for label, mine, theirs in checks:
        ok = theirs is not None and mine == theirs
        if not ok:
            bad.append(label)
        shown = mine if not isinstance(mine, str) or len(mine) < 46 else mine[:43] + "..."
        print(f"  {'OK ' if ok else 'BAD'} {label:16} {shown!r}")
        if not ok:
            print(f"      harness has: {theirs!r}")

    if bad:
        print(f"\nMISMATCH in {bad}.")
        print("The evaluated configuration and the shipped one have diverged, so the")
        print("reported WER no longer describes this application. Reconcile them.")
        return 1
    print("\napp/stt.py matches the evaluated configuration (malaysian_prompt_routed).")
    return 0


# ---------------------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(description="Transcribe code-switched Malay-English speech.")
    ap.add_argument("audio", nargs="?", help="path to a .wav file")
    ap.add_argument("--json", action="store_true", help="print language and timing too")
    ap.add_argument("--check-config", action="store_true",
                    help="verify settings match eval/speech/run_wer.py")
    ap.add_argument("--warm-up", action="store_true",
                    help="load both models, downloading only if absent; report the cache")
    args = ap.parse_args()

    if args.check_config:
        return check_config()
    if args.warm_up:
        # Report the cache before and after, so it is obvious which runs downloaded
        # anything and which merely read from disk.
        before = cache_status()
        started = time.perf_counter()
        warm_up()
        elapsed = time.perf_counter() - started

        total = 0
        for key, info in cache_status().items():
            total += info["bytes"]
            action = "downloaded" if not before[key]["cached"] else "already cached"
            print(f"  {info['name']}")
            print(f"    {human_bytes(info['bytes']):>8}  {action}")
            print(f"    {info['path']}")
        print(f"\n{human_bytes(total)} on disk, loaded in {elapsed:.1f}s.")
        return 0
    if not args.audio:
        ap.print_help()
        return 1

    result = transcribe_detailed(args.audio)
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print(result.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
