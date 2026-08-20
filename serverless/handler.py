#!/usr/bin/env python3
# RunPod Serverless handler for the vanKonijnenburg SaaS product. Same extract-then-compose-cite
# pipeline as scripts/model_server.py (the on-demand Pod version), adapted to RunPod's serverless
# worker model: the model loads ONCE at container start (module level, below), and RunPod's own
# runtime calls handler(job) per request on a warm worker -- no HTTP server, no pod lifecycle
# management, no idle-shutdown code needed, RunPod's serverless scheduler handles all of that.
import os, re, time

# Must happen before `import torch` -- see model_server.py for the full story: uncapped thread
# pools thrash badly when os.cpu_count() reads a shared host's full core count instead of the
# container's real cgroup CPU quota, making requests look permanently hung at 0% GPU utilization.
_TORCH_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "8"))
os.environ.setdefault("OMP_NUM_THREADS", str(_TORCH_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_TORCH_THREADS))

import runpod
import torch
torch.set_num_threads(_TORCH_THREADS)
torch.set_num_interop_threads(1)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

EVAL_SCRIPT = os.environ.get("EVAL_SCRIPT_PATH", "/app/eval_extract_compose_gemma.py")
MODEL = os.environ.get("MODEL", "google/gemma-4-31B-it")
ADAPTER = os.environ.get("ADAPTER_PATH", "/app/adapter")
CHUNK_CHARS = int(os.environ.get("CHUNK_CHARS", "6000"))

# Pull the exact, already-tested EXTRACT_SYSTEM / COMPOSE_SYSTEM prompt strings straight from the
# finalized eval script instead of copy-pasting them here, so this handler can never silently
# drift from the pipeline that was actually validated against FACTS-860 + HaluEval.
_ns = {}
with open(EVAL_SCRIPT, encoding="utf-8") as _f:
    _lines = _f.readlines()
_cutoff = next(i for i, l in enumerate(_lines) if l.startswith('print(f"=== Extract-then-compose'))
exec("".join(_lines[:_cutoff]), _ns)
EXTRACT_SYSTEM = _ns["EXTRACT_SYSTEM"]
COMPOSE_SYSTEM = _ns["COMPOSE_SYSTEM"]

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

_state = {"model": None, "tok": None, "stop_ids": None}


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
    _state.update(model=model, tok=tok, stop_ids=stop_ids)
    print("=== Model klaar ===", flush=True)


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
    # COMPOSE_SYSTEM's own documented output format is "REASONING: ... FINAL ANSWER: ...", but
    # nothing was ever stripping the REASONING scratchpad before this text got shown to users or
    # fed into the citation-matching pass -- confirmed live: real answers had a leaked "REASONING:"
    # block and a duplicated "**Final Answer:**" section (from a continuation round re-emitting
    # part of the marker) in front of the actual prose, which also broke every citation match since
    # the "sentences" being matched against the source document were really reasoning-list bullets.
    # Takes the LAST marker match specifically, since a continuation round can repeat it.
    _matches = list(re.finditer(r"\*{0,2}final answer\*{0,2}:?", answer, flags=re.IGNORECASE))
    if _matches:
        answer = answer[_matches[-1].end():].lstrip(" :\n")

    _leak = re.search(r"\bthought\b|\bwait\b[*_]{0,2}[,.\-—]|\blet me restart\b", answer, flags=re.IGNORECASE)
    if _leak and _leak.start() > 30:
        answer = answer[:_leak.start()].rstrip()
    _degenerate = re.search(r"(.{2,80}?)\1{3,}", answer, flags=re.DOTALL)
    if _degenerate and _degenerate.start() > 30:
        answer = answer[:_degenerate.start()].rstrip()
    return answer.strip()


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
    per_chunk_hits = [cite_chunk(c, numbered_claims) for c in chunks]
    citations = []
    for i, s in enumerate(sentences):
        quote = next((hits[i] for hits in per_chunk_hits if i in hits), None)
        citations.append({"claim": s, "quote": quote, "grounded": quote is not None})
    return citations


def analyze(question, document):
    t0 = time.time()
    chunks = split_into_chunks(document)
    extractions = [extract_chunk(question, c) for c in chunks]
    combined_facts = "\n\n".join(f"[Deel {i+1}/{len(chunks)}]\n{e}" for i, e in enumerate(extractions))

    compose_user = f"QUESTION: {question}\n\nFACTS LIST (extracted earlier):\n{combined_facts}"
    answer = clean_answer(gen(COMPOSE_SYSTEM, compose_user, max_new=700, max_continuations=1))
    citations = cite_answer(chunks, answer)

    return {
        "answer": answer,
        "citations": citations,
        "chunk_count": len(chunks),
        "seconds": round(time.time() - t0, 1),
    }


def handler(job):
    job_input = job.get("input") or {}
    question = str(job_input.get("question") or "").strip()
    document = str(job_input.get("document") or "")
    if not question:
        return {"error": "Missing 'question' in input."}
    return analyze(question, document)


load_model()
runpod.serverless.start({"handler": handler})
