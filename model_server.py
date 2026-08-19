#!/usr/bin/env python3
# Persistent inference server for the vanKonijnenburg SaaS product: wraps the same extract-then-
# compose pipeline used throughout the FACTS-860 eval work (scripts/eval_extract_compose_gemma.py)
# as a long-lived HTTP endpoint, so a RunPod pod loads the model ONCE and serves many documents
# instead of a cold model-load per request. Large ("complex") documents get split into chunks and
# extracted in parallel (one agent call per chunk), then composed together into one answer.
import os, re, time, threading
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

EVAL_SCRIPT = os.environ.get("EVAL_SCRIPT_PATH", "/workspace/hallucheck/eval_extract_compose_gemma.py")
MODEL = os.environ.get("MODEL", "google/gemma-4-31B-it")
ADAPTER = os.environ.get("ADAPTER_PATH", "/workspace/dpo_gemma31b_grounding-adapter_v2")
CHUNK_CHARS = int(os.environ.get("CHUNK_CHARS", "6000"))
MAX_WORKERS = int(os.environ.get("EXTRACT_WORKERS", "4"))

# Pull the exact, already-tested EXTRACT_SYSTEM / COMPOSE_SYSTEM prompt strings straight from the
# finalized eval script instead of copy-pasting them here, so this server can never silently drift
# from the pipeline that was actually validated against FACTS-860 + HaluEval.
_ns = {}
with open(EVAL_SCRIPT, encoding="utf-8") as _f:
    _lines = _f.readlines()
_cutoff = next(i for i, l in enumerate(_lines) if l.startswith('print(f"=== Extract-then-compose'))
exec("".join(_lines[:_cutoff]), _ns)
EXTRACT_SYSTEM = _ns["EXTRACT_SYSTEM"]
COMPOSE_SYSTEM = _ns["COMPOSE_SYSTEM"]

# Additive citation pass (NOT part of the frozen/tested extract-then-compose pipeline above --
# this only decorates its output for the product UI, so it can't affect the pipeline's own
# grounding accuracy). Pairs every sentence of the final answer with an exact quoted span from the
# source document, or flags it as unsupported -- this is the core trust feature of the product:
# a reviewer must be able to check each claim against the document in one click, since the model
# itself can still make reasoning errors or hallucinate.
#
# Runs per-chunk (same chunks the extraction step already produced) rather than against a single
# truncated slice of the document -- a long document's supporting passage can land anywhere, and
# truncating to the first ~8000 chars would silently fail to find citations for claims sourced
# from later pages. Claims are numbered ONCE in Python before fan-out, so results from independent
# per-chunk LLM calls can be merged by index instead of relying on the model to re-derive the same
# sentence split consistently across calls.
CITE_SYSTEM = (
    "You are a citation-checking assistant. You will be given ONE CHUNK of a larger document and a "
    "numbered list of CLAIMS taken from an answer written about the full document. For each claim, "
    "check ONLY whether THIS CHUNK contains an exact supporting quote for it -- the supporting text "
    "may be in a different chunk you don't see, and that is expected and fine. If this chunk supports "
    "a claim, copy the shortest exact substring from this chunk that proves it, character-for-"
    "character (never paraphrase). If this chunk does NOT support a claim, simply omit that claim's "
    "index from your output entirely -- do not guess or invent a quote.\n\n"
    "Reply with ONLY a JSON array, no other text, containing one entry per claim THIS CHUNK supports:\n"
    '[{"index": <claim number>, "quote": "<exact substring from this chunk>"}]'
)


def cite_chunk(chunk, numbered_claims):
    cite_user = f"DOCUMENT CHUNK:\n{chunk}\n\nCLAIMS:\n{numbered_claims}"
    raw = gen(CITE_SYSTEM, cite_user, max_new=800, max_continuations=1)
    match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if not match:
        return {}
    try:
        import json
        parsed = json.loads(match.group(0))
        return {
            int(item["index"]): str(item["quote"])
            for item in parsed
            if isinstance(item, dict) and item.get("quote") and "index" in item
        }
    except Exception:
        return {}


def cite_answer(chunks, answer):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return []
    numbered_claims = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(chunks))) as ex:
        per_chunk_hits = list(ex.map(lambda c: cite_chunk(c, numbered_claims), chunks))

    citations = []
    for i, s in enumerate(sentences):
        quote = next((hits[i] for hits in per_chunk_hits if i in hits), None)
        citations.append({"claim": s, "quote": quote, "grounded": quote is not None})
    return citations

app = FastAPI()
_model_lock = threading.Lock()
_state = {"ready": False, "model": None, "tok": None, "stop_ids": None}


def load_model():
    print(f"=== Model laden ({MODEL}) + adapter ({ADAPTER}) ===", flush=True)
    BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": 0}, dtype=torch.bfloat16, quantization_config=BNB)
    model = PeftModel.from_pretrained(base, ADAPTER) if ADAPTER else base
    eot_id = tok.convert_tokens_to_ids("<end_of_turn>")
    stop_ids = {tok.eos_token_id}
    if isinstance(eot_id, int) and eot_id >= 0:
        stop_ids.add(eot_id)
    _state.update(model=model, tok=tok, stop_ids=stop_ids, ready=True)
    print("=== Model klaar, server accepteert nu requests ===", flush=True)


def gen(system, user, max_new, max_continuations=1):
    model, tok, stop_ids = _state["model"], _state["tok"], _state["stop_ids"]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    full_text = ""
    for _round in range(1 + max_continuations):
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.15,
                                  eos_token_id=list(stop_ids), pad_token_id=tok.pad_token_id)
        new_ids = out[0][enc["input_ids"].shape[1]:]
        chunk = tok.decode(new_ids, skip_special_tokens=True).strip()
        full_text += (" " if full_text else "") + chunk
        ended_naturally = len(new_ids) == 0 or new_ids[-1].item() in stop_ids
        if ended_naturally:
            break
        messages.append({"role": "assistant", "content": chunk})
        messages.append({"role": "user", "content": "Ga door precies waar je gebleven was, zonder iets te herhalen."})
    return full_text.strip()


def clean_answer(answer):
    _leak = re.search(r"\bthought\b|\bwait\b[*_]{0,2}[,.\-—]|\blet me restart\b", answer, flags=re.IGNORECASE)
    if _leak and _leak.start() > 30:
        answer = answer[:_leak.start()].rstrip()
    _degenerate = re.search(r"(.{2,80}?)\1{3,}", answer, flags=re.DOTALL)
    if _degenerate and _degenerate.start() > 30:
        answer = answer[:_degenerate.start()].rstrip()
    return answer


def split_into_chunks(text, size=CHUNK_CHARS):
    paras = text.split("\n\n")
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 > size and cur:
            chunks.append(cur)
            cur = p
        else:
            cur = cur + "\n\n" + p if cur else p
    if cur:
        chunks.append(cur)
    return chunks or [text]


def extract_chunk(question, chunk):
    extract_user = f"QUESTION: {question}\n\nDOCUMENT:\n{chunk}"
    extraction = gen(EXTRACT_SYSTEM, extract_user, max_new=500, max_continuations=1)
    end = re.search(r"</self_described>", extraction, flags=re.IGNORECASE)
    return extraction[: end.end()] if end else extraction


class AnalyzeRequest(BaseModel):
    question: str
    document: str


@app.get("/health")
def health():
    return {"ok": _state["ready"]}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    t0 = time.time()
    chunks = split_into_chunks(req.document)
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(chunks))) as ex:
        extractions = list(ex.map(lambda c: extract_chunk(req.question, c), chunks))
    combined_facts = "\n\n".join(f"[Deel {i+1}/{len(chunks)}]\n{e}" for i, e in enumerate(extractions))

    compose_user = f"QUESTION: {req.question}\n\nFACTS LIST (extracted earlier):\n{combined_facts}"
    answer = clean_answer(gen(COMPOSE_SYSTEM, compose_user, max_new=700, max_continuations=1))
    citations = cite_answer(chunks, answer)

    return {
        "answer": answer,
        "citations": citations,
        "chunk_count": len(chunks),
        "extraction": combined_facts,
        "seconds": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
