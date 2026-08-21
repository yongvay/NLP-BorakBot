"""BorakBot — Streamlit interface.

CURRENT STATE: Stage 1 only. Record or upload speech, get rojak text back.

Stages 2-4 (normalisation, the fine-tuned model, feedback logging) are not built yet, so
this app deliberately stops at the transcript rather than faking a reply. The wiring points
are marked TODO below.

Starting the server costs ~13 s: importing torch and transformers, then reading ~930 MB of
weights off disk. Only the very first run downloads anything. DESIGN.md §4.

Usage
    pip install -r requirements.txt
    python app/stt.py --warm-up          # first time only, downloads ~930 MB
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import stt  # noqa: E402

st.set_page_config(page_title="BorakBot", page_icon="🎙", layout="centered")


@st.cache_resource(show_spinner=False)
def _load_models():
    """Cached across reruns, so the models load once per server, not once per click."""
    stt.warm_up()
    return True


st.title("BorakBot")
st.caption("Bahasa Rojak speech-to-text — Stage 1 of the pipeline")

# Load before the widgets rather than on the first clip. The wait is the same either way,
# but here it overlaps with the user reading the page instead of landing in the silence
# after they press record.
with st.spinner("Loading the models (~13 s, or a one-off ~930 MB download on first run)…"):
    _load_models()

with st.sidebar:
    st.subheader("Configuration")
    st.markdown(
        f"""
**Detector** `whisper-{stt.VANILLA_MODEL}`

**Transcriber** `{stt.MALAYSIAN_MODEL.split('/')[-1]}`
**English threshold** `{stt.EN_THRESHOLD}`

Measured WER on the 48-clip evaluation set: **0.349** strict, **0.339** lenient.
"""
    )
    st.divider()
    st.caption(
        "The detector only chooses the language token. The Malaysian model does all "
        "the transcribing either way."
    )

tab_record, tab_upload = st.tabs(["Record", "Upload a file"])
with tab_record:
    recorded = st.audio_input("Press record and speak in rojak")
with tab_upload:
    # Phone voice memos arrive as .mp4 or .3gp containers, not just .m4a. ffmpeg sniffs the
    # container and ignores the extension, so only this whitelist had to widen.
    uploaded = st.file_uploader(
        "WAV, MP3, M4A, MP4, 3GP or OGG",
        type=["wav", "mp3", "m4a", "mp4", "3gp", "ogg", "opus", "flac"],
    )

audio = recorded or uploaded

if audio is not None:
    with st.spinner("Transcribing..."):
        try:
            result = stt.transcribe_detailed(audio)
        except Exception as exc:                      # noqa: BLE001
            st.error(f"Transcription failed: {exc}")
            st.stop()

    st.subheader("Transcript")
    st.write(result.text if result.text else "_(nothing recognised)_")

    a, b, c = st.columns(3)
    a.metric("Language token", result.language)
    b.metric("P(english)", f"{result.p_en:.3f}")
    c.metric("Time", f"{result.seconds:.1f}s")

    if result.language == "ms" and result.p_en > 0.3:
        st.info(
            f"The detector gave this clip P(en) = {result.p_en:.3f}, below the "
            f"{stt.EN_THRESHOLD} threshold, so it was transcribed as Malay. Clips in "
            "this range are where routing is least reliable — see DESIGN.md §2."
        )

    # TODO Stage 2: normalised = normalise.clean(result.text)
    # TODO Stage 3: reply = inference.answer(normalised, history)
    # TODO Stage 4: thumbs up/down, correction box, log to SQLite
    st.divider()
    st.caption(
        "Next stages — normalisation, the fine-tuned model, and feedback logging — "
        "are not wired up yet."
    )
else:
    st.info("Record something or upload a file to begin.")
