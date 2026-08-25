"""The fine-tuned model — Stage 3 of the pipeline.

    normalised utterance + recent history + system prompt  ->  reply

Serves `meta-llama/Llama-3.2-3B-Instruct` with the round-1 QLoRA adapter
attached. Knowledge is in the weights: there is no retrieval step, and the
refusal behaviour is trained rather than filtered. What the adapter is worth,
measured on the 63-item test split, is in DESIGN.md §10 -- perplexity 23.75 ->
3.65, in-domain over-refusal 67% -> 2%.

Decoding is greedy and `max_new_tokens` is 160, both matching `eval/generate.py`
exactly. If they drift, the demo stops being the thing the report measured.

FOUR THINGS THAT ARE NOT OBVIOUS

1.  The system prompt is read from the knowledge base, not stored here.
    `data/knowledge_base/_system_prompt.md` is the same file
    `training/to_llamafactory.py` stamps onto every training example and that
    `eval/generate.py` parses. Three readers, one source: the prompt text cannot
    drift between training, evaluation and the demo. (Each reader parses it
    independently -- that is duplication of ten lines of parsing, not of the
    prompt itself, which is the part that matters.)

2.  History is capped at two exchanges, and that is generous.
    Every training example is a SINGLE turn: `to_llamafactory.py` emits
    `conversations: [human, gpt]` and nothing longer. The model has never seen a
    multi-turn context, so history is off-distribution by construction. Two
    exchanges is enough for "and how much is that?" to resolve, and short enough
    that the prompt still looks broadly like training. Raise HISTORY_TURNS and
    quality degrades before the context window is anywhere near full.

3.  This module may have to switch the Hub back online.
    `app/stt.py` sets `HF_HUB_OFFLINE=1` at import once the Whisper model is
    cached, to skip a revision check that stalls on bad wi-fi. That flag is
    global. On a machine where Whisper is cached but Llama is not, it would turn
    Llama's first download into an obscure offline error. The check below runs at
    import, before `huggingface_hub` is imported anywhere, because the library
    reads the variable into a module constant and never looks again.

4.  4-bit needs CUDA; the CPU path is a fallback, not a mode.
    bitsandbytes has no CPU kernel. Without a GPU this loads bfloat16 on CPU:
    ~6.4 GB of RAM and roughly a minute per reply. It runs, so the app is
    demonstrable on any machine, but it is not what DESIGN.md §10 measured and
    not what should be shown to a grader.

Setup
    hf auth login          # gated base model + private adapter
    python app/inference.py --warm-up
    python app/inference.py "eh macam mana nak renew roadtax"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------- settings

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER = "YongVay/borakbot-qlora-r1"

# Mirrored from eval/generate.py. Changing either invalidates DESIGN.md §10 as a
# description of what the demo does.
MAX_NEW_TOKENS = 160
GREEDY = True

# See note 2 in the module docstring. One "exchange" is a user turn plus a reply.
HISTORY_TURNS = 2

ROOT = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT_MD = ROOT / "data" / "knowledge_base" / "_system_prompt.md"

_models: dict = {}


# ------------------------------------------------------------- hub availability
# Note 3. Must run at import, before huggingface_hub is imported anywhere.

def _repo_is_cached(repo_id: str) -> bool:
    hub = Path(os.getenv("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"
    return (hub / f"models--{repo_id.replace('/', '--')}").is_dir()


if os.environ.get("HF_HUB_OFFLINE") == "1" and not _repo_is_cached(BASE_MODEL):
    # stt.py went offline for Whisper's benefit; this model still has to arrive.
    os.environ.pop("HF_HUB_OFFLINE")


# ------------------------------------------------------------------- the prompt

def load_system_prompt(path: Path = SYSTEM_PROMPT_MD) -> str:
    """Extract the fenced prompt under '## Draft v1'.

    The file is prose with the prompt embedded in it, because its reasoning is
    examinable material. Same parse as eval/generate.py, same source file.
    """
    text = path.read_text(encoding="utf-8")
    after = text.split("## Draft v1", 1)
    if len(after) != 2:
        raise RuntimeError(f"{path}: no '## Draft v1' heading")
    m = re.search(r"```\s*\n(.*?)\n```", after[1], re.S)
    if not m:
        raise RuntimeError(f"{path}: no fenced block under '## Draft v1'")
    return m.group(1).strip()


# ------------------------------------------------------------------ model access

def _load():
    """Base model in 4-bit where possible, adapter attached. Cached per process."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    four_bit = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    kwargs: dict = {}
    if four_bit:
        from transformers import BitsAndBytesConfig

        # Identical to eval/generate.py's --4bit configuration.
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    else:
        # Note 4. bfloat16 rather than float32: 6.4 GB instead of 12.8 GB, which
        # is the difference between slow and unloadable on a 16 GB laptop.
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kwargs)

    from peft import PeftModel

    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, model, four_bit


def _model():
    if "llm" not in _models:
        _models["llm"] = _load()
    return _models["llm"]


def warm_up() -> None:
    """Load the model. Call at application start, not on the first message."""
    _model()


def is_loaded() -> bool:
    return "llm" in _models


def backend() -> str:
    """'cuda-4bit' or 'cpu-bf16'. For the sidebar, and for knowing which one the
    demo actually ran on when the latency is questioned afterwards."""
    import torch

    return "cuda-4bit" if torch.cuda.is_available() else "cpu-bf16"


# ------------------------------------------------------------------- generation

@dataclass
class Reply:
    text: str
    seconds: float
    backend: str


def build_messages(user: str, history: list[dict] | None = None) -> list[dict]:
    """System prompt, the last HISTORY_TURNS exchanges, then the new turn.

    `history` is a list of {"role": "user"|"assistant", "content": str} in
    chronological order -- Streamlit's own message format, so the caller can
    pass its session state straight in.
    """
    msgs = [{"role": "system", "content": load_system_prompt()}]
    if history:
        # Two exchanges = four messages. Slicing messages rather than pairs keeps
        # this correct even if the last reply failed and the history is odd.
        msgs.extend(history[-HISTORY_TURNS * 2:])
    msgs.append({"role": "user", "content": user})
    return msgs


def answer(user: str, history: list[dict] | None = None) -> Reply:
    """Stage 3. Takes the NORMALISED utterance; see app/normalise.py."""
    import torch

    tok, model, four_bit = _model()
    started = time.perf_counter()

    text = tok.apply_chat_template(
        build_messages(user, history), tokenize=False, add_generation_prompt=True
    )
    enc = tok(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=not GREEDY,
            pad_token_id=tok.pad_token_id,
        )

    reply = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return Reply(
        text=reply,
        seconds=time.perf_counter() - started,
        backend="cuda-4bit" if four_bit else "cpu-bf16",
    )


# -------------------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(description="Ask the fine-tuned BorakBot a question.")
    ap.add_argument("text", nargs="?", help="a normalised rojak utterance")
    ap.add_argument("--warm-up", action="store_true",
                    help="load the model, downloading it if absent")
    ap.add_argument("--show-prompt", action="store_true",
                    help="print the system prompt and exit; no model load")
    args = ap.parse_args()

    if args.show_prompt:
        print(load_system_prompt())
        return 0

    if args.warm_up:
        started = time.perf_counter()
        warm_up()
        print(f"{backend()} ready in {time.perf_counter() - started:.1f}s")
        return 0

    if not args.text:
        ap.print_help()
        return 1

    result = answer(args.text)
    print(result.text)
    print(f"\n[{result.backend}, {result.seconds:.1f}s]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
