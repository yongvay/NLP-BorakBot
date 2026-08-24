"""BorakBot — Streamlit interface.

CURRENT STATE: Stages 1-3. Speak or type rojak, get a rojak answer back.

    speech -> Whisper (stt) -> ASR repair (normalise) -> fine-tuned LLM (inference)

Stage 4, the thumbs-up/down feedback loop into SQLite, is not built yet; the wiring
point is marked TODO at the bottom of the reply block.

Starting the server is slow and front-loaded on purpose: ~13 s for Whisper, then the
LLM on top of it -- seconds on a GPU, a minute or two on CPU while ~6 GB of weights
come off disk. Doing it before the widgets means the wait overlaps with the user
reading the page instead of landing after they press record. DESIGN.md §4.

Usage
    pip install -r requirements.txt
    huggingface-cli login                    # gated base model + private adapter
    python app/stt.py --warm-up              # first time only, ~930 MB
    python app/inference.py --warm-up        # first time only, ~6 GB
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import inference, normalise, stt  # noqa: E402

st.set_page_config(page_title="BorakBot", page_icon="🎙", layout="centered")


@st.cache_resource(show_spinner=False)
def _load_models():
    """Cached across reruns, so the models load once per server, not once per click."""
    stt.warm_up()
    inference.warm_up()
    return True


st.title("BorakBot")
st.caption("Bahasa Rojak chatbot — speech in, rojak out")

with st.spinner("Loading Whisper and the fine-tuned model (first run downloads ~7 GB)…"):
    _load_models()

# ------------------------------------------------------------------------ sidebar

with st.sidebar:
    st.subheader("Configuration")
    st.markdown(
        f"""
**Transcriber** `{stt.MALAYSIAN_MODEL.split('/')[-1]}`
**Detector** `whisper-{stt.VANILLA_MODEL}`
**English threshold** `{stt.EN_THRESHOLD}`

**Model** `{inference.BASE_MODEL.split('/')[-1]}`
**Adapter** `{inference.ADAPTER.split('/')[-1]}`
**Backend** `{inference.backend()}`
"""
    )
    if inference.backend() == "cpu-bf16":
        st.warning(
            "Running the LLM on CPU — expect roughly a minute per reply. Install a "
            "CUDA build of torch plus bitsandbytes for the 4-bit GPU path."
        )
    st.divider()
    st.markdown(
        """
WER on the 48-clip evaluation set: **0.349** strict.
Adapter vs base on the 63-item test split: perplexity **23.75 → 3.65**,
wrongly-declined in-domain questions **67% → 2%**. DESIGN.md §10.
"""
    )
    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------- chat history

if "messages" not in st.session_state:
    st.session_state.messages = []
# Streamlit re-runs the whole script on every interaction and hands back the same
# recording each time. Without remembering which clip has already been answered, every
# rerun would re-transcribe and re-generate it.
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


def respond(user_text: str, transcript: str | None = None) -> None:
    """Run Stages 2-3 on one utterance and append both turns to the transcript."""
    cleaned = normalise.normalise(user_text)

    with st.chat_message("user"):
        st.write(cleaned.text)
        if transcript is not None:
            st.caption(f"heard: _{transcript}_")
        if cleaned.repairs:
            fixes = ", ".join(f"{a} → {b}" for a, b in cleaned.repairs)
            st.caption(f"repaired: {fixes}")

    # History excludes the turn being answered; build_messages appends it.
    history = st.session_state.messages[:]
    st.session_state.messages.append({"role": "user", "content": cleaned.text})

    with st.chat_message("assistant"):
        with st.spinner("Fikir kejap…"):
            try:
                reply = inference.answer(cleaned.text, history)
            except Exception as exc:                      # noqa: BLE001
                st.error(f"Generation failed: {exc}")
                return
        st.write(reply.text)
        st.caption(f"{reply.backend}, {reply.seconds:.1f}s")

    st.session_state.messages.append({"role": "assistant", "content": reply.text})

    # TODO Stage 4: thumbs up/down, correction box on thumbs-down, log the whole
    # exchange to SQLite with the conversation context.


# ------------------------------------------------------------------------- input

typed = st.chat_input("Tanya apa-apa — roadtax, MyKad, passport, atau slang")

with st.expander("🎙 Or speak instead"):
    tab_record, tab_upload = st.tabs(["Record", "Upload a file"])
    with tab_record:
        recorded = st.audio_input("Press record and speak in rojak")
    with tab_upload:
        # Phone voice memos arrive as .mp4 or .3gp containers, not just .m4a. ffmpeg
        # sniffs the container and ignores the extension, so only this list had to widen.
        uploaded = st.file_uploader(
            "WAV, MP3, M4A, MP4, 3GP or OGG",
            type=["wav", "mp3", "m4a", "mp4", "3gp", "ogg", "opus", "flac"],
        )

audio = recorded or uploaded

if typed:
    respond(typed)

elif audio is not None:
    payload = audio.getvalue()
    if payload != st.session_state.last_audio:
        st.session_state.last_audio = payload
        with st.spinner("Transcribing…"):
            try:
                result = stt.transcribe_detailed(payload)
            except Exception as exc:                      # noqa: BLE001
                st.error(f"Transcription failed: {exc}")
                st.stop()

        if not result.text.strip():
            st.warning("Nothing recognised in that clip — try again.")
        else:
            respond(result.text, transcript=result.text)

elif not st.session_state.messages:
    st.info("Type a question, or open the recorder to speak one.")
