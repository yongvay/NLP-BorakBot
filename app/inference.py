"""The fine-tuned model — Stage 3 of the pipeline.

    normalised utterance + recent history + system prompt  ->  reply

Serves `meta-llama/Llama-3.2-3B-Instruct` with the round-1 QLoRA adapter
attached. Knowledge is in the weights: there is no retrieval step, and the
refusal behaviour is trained rather than filtered. What the adapter is worth,
measured on the 63-item test split, is in the Part B report -- perplexity 23.75 ->
3.65, in-domain over-refusal 67% -> 2%.

Decoding is greedy and `max_new_tokens` is 160, both matching `eval/generate.py`
exactly. If they drift, the demo stops being the thing the report measured.

FIVE THINGS THAT ARE NOT OBVIOUS

1.  The system prompt is read from the knowledge base, not stored here.
    `stage2_chatbot/knowledge_base/_system_prompt.md` is the same file
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
    demonstrable on any machine, but it is not the configuration the report measured and
    not what should be shown to a grader.

5.  The model is pinned to the GPU: it goes there whole, or not at all.
    `device_map={"": 0}` rather than `"auto"`. On a 4 GB card the fit is ~3.0 GB
    against 4.0 GB total, and `"auto"` does NOT fail when that comes up short --
    it quietly spills layers into system RAM. The app would then report
    `cuda-4bit` while running at a fraction of GPU speed, which is the one
    failure that looks like success. Pinning turns a short card into an
    OutOfMemoryError, caught below and reported honestly.

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

# Mirrored from eval/generate.py. Changing either invalidates the reported figures
# as a description of what the demo does.
MAX_NEW_TOKENS = 160
GREEDY = True

# See note 2 in the module docstring. One "exchange" is a user turn plus a reply.
HISTORY_TURNS = 2

ROOT = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT_MD = ROOT / "stage2_chatbot" / "knowledge_base" / "_system_prompt.md"

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

    def cpu_bf16():
        # Note 4. bfloat16 rather than float32: 6.4 GB instead of 12.8 GB, which
        # is the difference between slow and unloadable on a 16 GB laptop.
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map=None
        )

    if four_bit:
        from transformers import BitsAndBytesConfig

        # Identical to eval/generate.py's --4bit configuration.
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        try:
            # Note 5. Pin, do not spill.
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, quantization_config=quant, device_map={"": 0}
            )
        except Exception as exc:
            # Deliberately broad. The GPU path is an optimisation, and ANY way it
            # fails should end up on the path that always works rather than
            # halfway through a demo. Two real ones seen here: OutOfMemoryError
            # when something else is holding the card, and a bitsandbytes
            # RuntimeError ("CPU-only version of bitsandbytes") when torch
            # reports a GPU that bnb cannot actually use. Catching only the
            # former left the latter fatal.
            print(f"4-bit GPU load failed ({type(exc).__name__}: {exc}); "
                  f"falling back to CPU bfloat16.", file=sys.stderr)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            # `four_bit` is what backend() reports, so the sidebar tells the
            # truth about which path the demo actually ran on.
            four_bit = False
            model = cpu_bf16()
    else:
        model = cpu_bf16()

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


def cuda_present() -> bool:
    """Whether the machine HAS a usable card, regardless of what got loaded on it.

    Distinguishes the two ways backend() can say 'cpu-bf16': no GPU at all, or a
    GPU whose 4-bit load did not fit (note 5). They need opposite advice, so the
    sidebar asks this before telling anyone what to go and install.
    """
    import torch

    return torch.cuda.is_available()


def backend() -> str:
    """'cuda-4bit' or 'cpu-bf16'. For the sidebar, and for knowing which one the
    demo actually ran on when the latency is questioned afterwards.

    Reports what actually loaded once something has. Probing CUDA instead would
    be wrong in exactly the case worth knowing about: a machine with a GPU whose
    load fell back to the CPU anyway (note 5).
    """
    import torch

    if is_loaded():
        return "cuda-4bit" if _models["llm"][2] else "cpu-bf16"
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
