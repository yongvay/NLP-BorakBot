"""Feedback review — the admin half of Stage 4.

Everything `python app/feedback.py` does from the terminal, done by clicking.
The demo should not require a second window and a typed command to show that
corrections are being captured.

WHAT THE PASSWORD IS AND IS NOT

It is a **demo gate**, not authentication. It stops someone walking up to the
laptop mid-demo and clicking into the review page; it stops nothing else. The
database sits unencrypted beside the app, Streamlit serves this over plain HTTP
on localhost, and the password is compared in one line. Calling it security
would be overclaiming, and the honest version is the one to say out loud if
asked.

The password lives in `.streamlit/secrets.toml`, which is gitignored;
`.streamlit/secrets.toml.example` is committed so a fresh clone knows what to
create.

WHY THIS IS A SEPARATE PAGE

It imports `app.feedback` and nothing else. `app/streamlit_app.py` loads Whisper
and a 3B model, which is ~7 GB and two to four minutes; this page opens
instantly because it never touches either. Reviewing corrections has no reason
to wait for a language model to load.
"""

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app import feedback  # noqa: E402

st.set_page_config(page_title="BorakBot — review", page_icon="🗂", layout="wide")


def _authenticated() -> bool:
    """One-line password check against secrets.toml. See the docstring."""
    if st.session_state.get("admin_ok"):
        return True

    st.title("Feedback review")
    st.caption("Enter the review password to continue.")

    try:
        expected = st.secrets["admin_password"]
    except (KeyError, FileNotFoundError):
        st.error(
            "No `admin_password` in `.streamlit/secrets.toml`. Copy "
            "`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` "
            "and set one."
        )
        return False

    with st.form("login"):
        entered = st.text_input("Password", type="password")
        if st.form_submit_button("Open"):
            if entered == expected:
                st.session_state.admin_ok = True
                st.rerun()
            else:
                st.error("Wrong password.")
    return False


if not _authenticated():
    st.stop()

# ---------------------------------------------------------------------- the page

st.title("Feedback review")
st.caption(
    "Corrections logged by users, awaiting team review. Nothing here is applied "
    "to the model automatically — DESIGN.md §14."
)

entries = feedback.recent(limit=500)
tally = feedback.counts()
corrections = feedback.export_corrections()

a, b, c, d = st.columns(4)
a.metric("Total feedback", tally["up"] + tally["down"])
b.metric("👍 Correct", tally["up"])
c.metric("👎 Wrong", tally["down"])
d.metric("Corrections to review", len(corrections))

st.divider()

if not entries:
    st.info("Nothing logged yet. Use the chat page, then rate a reply.")
    st.stop()

# --------------------------------------------------------------------- the table

st.subheader("All feedback")

only = st.radio(
    "Show", ["Everything", "👍 only", "👎 only"], horizontal=True, label_visibility="collapsed"
)
wanted = {"Everything": None, "👍 only": "up", "👎 only": "down"}[only]
shown = [e for e in entries if wanted is None or e.verdict == wanted]

st.dataframe(
    [
        {
            "id": e.id,
            "when (UTC)": e.created_utc.replace("T", " ").replace("+00:00", ""),
            "verdict": "👍" if e.verdict == "up" else "👎",
            "question": e.user_text,
            # Blank when typed. A populated cell means the question arrived by
            # voice, so a bad answer may be an ASR failure rather than an NLP one.
            "heard by Whisper": e.transcript or "",
            "reply": e.reply,
            "should have been": e.correction or "",
            "backend": e.backend or "",
            "sec": round(e.seconds, 1) if e.seconds else None,
        }
        for e in shown
    ],
    width="stretch",
    hide_index=True,
)
st.caption(f"{len(shown)} of {len(entries)} entries.")

st.divider()

# ------------------------------------------------------------------ the download

st.subheader("Export corrections")

if not corrections:
    st.info("No corrections yet. They appear here when a user rates a reply 👎 and "
            "supplies the answer they expected.")
else:
    st.download_button(
        f"⬇ Download {len(corrections)} correction(s) as JSON",
        data=json.dumps(corrections, ensure_ascii=False, indent=2),
        file_name="corrections.json",
        mime="application/json",
        type="primary",
    )
    with st.expander("Preview"):
        st.json(corrections)

st.warning(
    "**Review before appending.** Not every thumbs-down is a real gap. A user who "
    "marks a *correct* refusal as wrong — asking the bot to explain photosynthesis, "
    "say — is asking it to answer outside its knowledge, and training on that would "
    "undo the hallucination control the model was fine-tuned for. That judgement is "
    "the reason this step is manual."
)

st.divider()

# ------------------------------------------------------------- the retrain recipe

st.subheader("How these become a better model")
st.caption("Shown, not run — steps 4-6 write to disk and step 6 needs a Colab GPU.")

st.markdown(
    """
1. **Download** the corrections above.
2. **Review by hand** — reject anything outside the knowledge boundary.
3. **Add corpus metadata** to the survivors (`domain`, `form`, `source_fact`,
   `switch_type`, `batch`) and append them to a new `pairs_v2.jsonl`.
   The exporter cannot infer which knowledge-base fact a correction belongs to.
"""
)
st.code(
    "python scripts/split_corpus.py --in data/generated/pairs_v2.jsonl\n"
    "python training/to_llamafactory.py",
    language="powershell",
)
st.markdown(
    """
4. **Re-train** with `training/colab_finetune.ipynb`, pointing `output_dir` at a
   round-2 directory.
5. **Re-evaluate** with `training/colab_evaluate.ipynb` using `--tag tuned_v2`,
   scored against `tuned` on the same 63-item test split. That comparison is the
   evidence of improvement over time.
"""
)