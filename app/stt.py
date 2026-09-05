"""Speech-to-text for BorakBot — Stage 1 of the pipeline.

Two models, one pass each:

    whisper-small (vanilla)   ->  pick the language token           (detect_language)
    mesolitica/...-v3         ->  transcribe, given that token      (transcribe_detailed)

This is the `malaysian_prompt_routed` configuration, the best of the nine that
`stage1_stt/run_wer.py` measured: 0.349 strict WER, 0.339 lenient, over 48 clips
and 387 reference words. Why it is built this way — the two models, the threshold, the
prompt — is in the Part B report. Read that before the demo, not this file.

The settings below are duplicated from the evaluation harness rather than imported, because
importing it would drag pandas and jiwer into a Streamlit deployment that has no use for
them. Both files are now frozen: the harness has run and its results are committed. If you
ever change a constant here, the reported 0.349 stops describing what the demo runs.

Usage
    python app/stt.py recording.wav          # transcribe a file
    python app/stt.py recording.wav --json   # with language and timing
    python app/stt.py --warm-up              # download and cache the models

    from app.stt import transcribe
    text = transcribe("recording.wav")

If a first download is interrupted, the cache is left holding config.json without the
weights and loading raises OSError. Delete the model directory under
~/.cache/huggingface/hub/ and run --warm-up again.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Union

# --------------------------------------------------------------------- settings
# Mirrored from stage1_stt/run_wer.py. See the module docstring.

VANILLA_MODEL = "small"
MALAYSIAN_MODEL = "mesolitica/malaysian-whisper-small-v3"

# English is only chosen when the detector is confident, because the two failure directions
# are not symmetric.
EN_THRESHOLD = 0.5

# Transcribe on the CPU even when a GPU is present. The demo laptop has 4 GB of VRAM and
# Stage 3 needs ~3 GB of it for the 4-bit LLM; this model would take another ~1 GB and the
# two together do not fit. Whisper-small on CPU costs a few seconds, which is invisible
# beside generation.
#
# This does NOT affect the reported 0.349 WER: torch_dtype is float32 on either device, so
# the transcript is identical. Set BORAKBOT_STT_GPU=1 to override on a larger card.
STT_ON_GPU = os.getenv("BORAKBOT_STT_GPU") == "1"

# Guards against a decode that never terminates.
#
# Whisper emits <|endoftext|> when the audio gives it a reason to. Silence, room noise or
# a half-second button-press give it none, so the decoder loops on whatever it last
# produced until it hits its length ceiling -- 448 tokens by default, which on the CPU this
# stage is pinned to costs tens of seconds for a clip that should take two. It is the same
# unbounded decode behind the runaway diagnostic in the report (6.483 WER on one clip,
# 362 insertions against a seven-word reference).
#
# Two guards, because they fail differently. MIN_SECONDS and MIN_RMS keep non-speech away
# from both models; MAX_NEW_TOKENS bounds whatever does reach the transcriber.
#
# Neither can fire on the 48 evaluation clips -- every one is recorded speech far above the
# energy floor, and the longest hypothesis it produces is 10 words against a 128-token
# ceiling -- so
# the reported 0.349 is unchanged. MAX_NEW_TOKENS is mirrored in run_wer.py for the reason
# the module docstring gives; the two energy floors are not, because the harness must
# transcribe every clip in the manifest rather than skip any.
MIN_SECONDS = 0.4        # shorter than any real utterance
MIN_RMS = 0.005          # quieter than a spoken word in a quiet room
MAX_NEW_TOKENS = 128     # ~8x the longest utterance in the evaluation set
SAMPLE_RATE = 16_000     # what whisper.load_audio returns

# Decoder context. Itself code-switched, and sharing no vocabulary with the eval set.
ROJAK_PROMPT = (
    "cuaca panas gila hari ni kan, tak larat nak keluar. "
    "can you send me the file later, i need to check something first. "
    "eh awak dah reply that email ke belum lah. "
    "so how ah, do you want to join us or not."
)

AudioInput = Union[str, Path, bytes, IO[bytes]]

_models: dict = {}

# Skip the Hub's revision check once the model is on disk: a wasted second on a good
# network, a stall on a bad one, and campus wi-fi during the demo is the wrong place to
# find out which. This must run before huggingface_hub is imported anywhere -- it reads the
# variable into a module constant at import time and never looks at it again.
_HF_HUB = Path(os.getenv("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"
if (_HF_HUB / f"models--{MALAYSIAN_MODEL.replace('/', '--')}").is_dir():
    os.environ["HF_HUB_OFFLINE"] = "1"


# ------------------------------------------------------------------ model access

def _vanilla():
    """Base Whisper. Used only as a language detector, never to transcribe."""
    if "vanilla" not in _models:
        import whisper

        _models["vanilla"] = whisper.load_model(VANILLA_MODEL)
    return _models["vanilla"]


def _malaysian():
    """The Malaysian fine-tune. Does all the actual transcription."""
    if "malaysian" not in _models:
        import torch
        from transformers import pipeline

        _models["malaysian"] = pipeline(
            "automatic-speech-recognition",
            model=MALAYSIAN_MODEL,
            device=0 if STT_ON_GPU and torch.cuda.is_available() else -1,
            torch_dtype=torch.float32,
        )
    return _models["malaysian"]


def warm_up() -> None:
    """Load both models. Call once at application start, so the first user to press
    record does not sit through ~13 s of loading and assume the app has hung."""
    _vanilla()
    _malaysian()


# ------------------------------------------------------------------------- input

def _as_path(audio: AudioInput) -> tuple[str, str | None]:
    """Normalise any accepted input to a file path. Returns (path, path_to_delete).

    Whisper shells out to ffmpeg and needs a real file, but Streamlit's recorder hands back
    an in-memory buffer. The caller deletes the temporary file; letting it fall out of scope
    would delete it too early on Windows, where an open handle blocks the read.
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
    return tmp.name, tmp.name


def _has_speech(path: str) -> bool:
    """False when a clip is too short or too quiet to hold an utterance.

    Runs before either model, so a clip that fails here costs neither the encoder pass nor
    the runaway decode that non-speech audio provokes. Deliberately crude: this is a gate
    on obvious non-speech, not voice activity detection, and anything it passes is still
    bounded by MAX_NEW_TOKENS.
    """
    import numpy as np
    import whisper

    wav = whisper.load_audio(path)
    if wav.size / SAMPLE_RATE < MIN_SECONDS:
        return False
    return float(np.sqrt(np.mean(wav.astype("float64") ** 2))) >= MIN_RMS


# -------------------------------------------------------------------- the stages

def detect_language(audio: AudioInput) -> tuple[str, float]:
    """Choose the language token for an utterance. Returns (choice, p_en).

    Mesolitica's own detector always answers Malay, so English clips come back translated.
    Base Whisper's is unbiased and costs one encoder pass.

    p_en is returned alongside the decision because the decision alone cannot distinguish a
    clip missed narrowly from one missed completely.
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
        if tmp:
            Path(tmp).unlink(missing_ok=True)


@dataclass
class Transcription:
    text: str
    language: str          # token handed to the transcriber: "en", "ms", or "" if none
    p_en: float            # detector confidence, for logging and debugging
    seconds: float         # wall-clock, for the demo's latency discussion


def transcribe_detailed(audio: AudioInput) -> Transcription:
    """Full pipeline, returning the routing decision alongside the text."""
    started = time.perf_counter()
    path, tmp = _as_path(audio)
    try:
        if not _has_speech(path):
            # Empty text, which streamlit_app.py already reports as "nothing recognised".
            # language is "" because no token was ever handed to a transcriber.
            return Transcription(text="", language="", p_en=0.0,
                                 seconds=round(time.perf_counter() - started, 2))

        lang, p_en = detect_language(path)
        asr = _malaysian()

        # Mesolitica pins the Malay token in generation_config, which is why omitting
        # `language=` produced byte-identical output to forcing Malay. Passing it
        # explicitly is what makes the routed choice take effect.
        prompt_ids = asr.tokenizer.get_prompt_ids(ROJAK_PROMPT, return_tensors="pt")
        kwargs = {
            "task": "transcribe",
            "language": lang,
            "prompt_ids": prompt_ids.to(asr.model.device),
            # Ceiling on the decode, not on the prompt: max_new_tokens excludes the ~40
            # prompt tokens that max_length would count. Whisper will use all 448 of its
            # default budget on audio it cannot resolve.
            "max_new_tokens": MAX_NEW_TOKENS,
        }

        text = asr(path, generate_kwargs=kwargs)["text"].strip()
        return Transcription(text=text, language=lang, p_en=p_en,
                             seconds=round(time.perf_counter() - started, 2))
    finally:
        if tmp:
            Path(tmp).unlink(missing_ok=True)


def transcribe(audio: AudioInput) -> str:
    """Audio in, rojak text out. The function the rest of the app calls."""
    return transcribe_detailed(audio).text


# ---------------------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(description="Transcribe code-switched Malay-English speech.")
    ap.add_argument("audio", nargs="?", help="path to an audio file")
    ap.add_argument("--json", action="store_true", help="print language and timing too")
    ap.add_argument("--warm-up", action="store_true",
                    help="load both models, downloading them if absent")
    args = ap.parse_args()

    if args.warm_up:
        started = time.perf_counter()
        warm_up()
        print(f"both models ready in {time.perf_counter() - started:.1f}s")
        return 0
    if not args.audio:
        ap.print_help()
        return 1

    result = transcribe_detailed(args.audio)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2) if args.json
          else result.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
