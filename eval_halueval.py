import torch, json, os, re, time
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

_ns = {}
with open("/workspace/eval_extract_compose_gemma.py", encoding="utf-8") as _f:
    _all_lines = _f.readlines()
_cutoff = next(i for i, l in enumerate(_all_lines) if l.startswith('print(f"=== Extract-then-compose'))
exec("".join(_all_lines[:_cutoff]), _ns)
EXTRACT_SYSTEM = _ns["EXTRACT_SYSTEM"]
COMPOSE_SYSTEM = _ns["COMPOSE_SYSTEM"]
MAX_NEW_EXTRACT = int(os.environ.get("MAX_NEW_EXTRACT", "500"))
MAX_NEW_COMPOSE = int(os.environ.get("MAX_NEW_COMPOSE", "600"))
N = int(os.environ.get("N", "20"))
LABEL = os.environ.get("LABEL", "halueval")

MODEL = os.environ.get("MODEL", "google/gemma-4-31B-it")
ADAPTER = os.environ.get("ADAPTER_PATH", "/workspace/dpo_gemma31b_grounding-adapter")
USE_4BIT = os.environ.get("USE_4BIT", "0") == "1"
BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")

print("=== Data laden (HaluEval QA) ===", flush=True)
data = load_dataset("pminervini/HaluEval", "qa")["data"]
print("Kolommen:", data.column_names, flush=True)
print("Voorbeeld[0]:", {k: str(v)[:200] for k, v in data[0].items()}, flush=True)

# "echt uber moeilijk": langste knowledge-passage EN langste hallucinated_answer
# (meer info om in de war te raken, meer ruimte voor het model om subtiel af te dwalen)
OFFSET = int(os.environ.get("OFFSET", "0"))
sorted_rows = list(data)
sorted_rows.sort(key=lambda r: len(r.get("knowledge", "")) + len(r.get("hallucinated_answer", "")), reverse=True)
if "IDX_LIST" in os.environ:
    _idx_list = [int(x) for x in os.environ["IDX_LIST"].split(",")]
    rows = [(i, sorted_rows[i]) for i in _idx_list]
    print(f"=== Geselecteerd: exacte idx {_idx_list} uit de gesorteerde moeilijkste-lijst ===", flush=True)
else:
    rows = list(enumerate(sorted_rows))[OFFSET:OFFSET + N]
    print(f"=== Geselecteerd: rang {OFFSET}-{OFFSET+len(rows)} van moeilijkste items (langste knowledge+hallucinated_answer) ===", flush=True)

print(f"=== Model laden ({MODEL}) + adapter ({ADAPTER}) ===", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": 0}, dtype=torch.bfloat16, quantization_config=BNB if USE_4BIT else None)
model = PeftModel.from_pretrained(base, ADAPTER) if ADAPTER else base

_eot_id = tok.convert_tokens_to_ids("<end_of_turn>")
_stop_ids = {tok.eos_token_id}
if isinstance(_eot_id, int) and _eot_id >= 0:
    _stop_ids.add(_eot_id)

def gen(system, user, max_new, max_continuations=0):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    full_text = ""
    for _round in range(1 + max_continuations):
        _t = time.time()
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                  repetition_penalty=1.15,
                                  eos_token_id=list(_stop_ids), pad_token_id=tok.pad_token_id)
        new_ids = out[0][enc["input_ids"].shape[1]:]
        chunk = tok.decode(new_ids, skip_special_tokens=True).strip()
        full_text += (" " if full_text else "") + chunk
        ended_naturally = len(new_ids) == 0 or new_ids[-1].item() in _stop_ids
        print(f"    [ronde {_round+1}] {len(new_ids)} tokens, ended_naturally={ended_naturally}, tijd={time.time()-_t:.0f}s", flush=True)
        if ended_naturally:
            break
        messages.append({"role": "assistant", "content": chunk})
        messages.append({"role": "user", "content": "Ga door precies waar je gebleven was, zonder iets te herhalen."})
    return full_text.strip()

results = []
for i, (real_idx, row) in enumerate(rows):
    doc = row.get("knowledge", "")
    q = row.get("question", "")
    right = row.get("right_answer", "")
    halluc = row.get("hallucinated_answer", "")
    _t0 = time.time()

    extract_user = f"QUESTION: {q}\n\nDOCUMENT:\n{doc}"
    extraction = gen(EXTRACT_SYSTEM, extract_user, MAX_NEW_EXTRACT, max_continuations=1)
    _first_block_end = re.search(r"</self_described>", extraction, flags=re.IGNORECASE)
    if _first_block_end:
        extraction = extraction[:_first_block_end.end()]

    compose_user = f"QUESTION: {q}\n\nFACTS LIST (extracted earlier):\n{extraction}"
    answer = gen(COMPOSE_SYSTEM, compose_user, MAX_NEW_COMPOSE, max_continuations=1)

    _leak = re.search(r"\bthought\b|\bwait\b[*_]{0,2}[,.\-—]|\blet me restart\b", answer, flags=re.IGNORECASE)
    if _leak and _leak.start() > 30:
        answer = answer[:_leak.start()].rstrip()
    _degenerate = re.search(r"(.{2,80}?)\1{3,}", answer, flags=re.DOTALL)
    if _degenerate and _degenerate.start() > 30:
        answer = answer[:_degenerate.start()].rstrip()
    _tail_symbols = re.search(r"\n\s*[^\w\s]{1,10}\s*$", answer)
    while _tail_symbols and _tail_symbols.start() > 30:
        answer = answer[:_tail_symbols.start()].rstrip()
        _tail_symbols = re.search(r"\n\s*[^\w\s]{1,10}\s*$", answer)
    _tail_fragment = re.search(r"([.!?])\s*\n+\s*([^\n]{1,40})$", answer)
    if _tail_fragment and _tail_fragment.start(1) > 30 and not re.search(r"[.!?]\s*$", _tail_fragment.group(2)):
        answer = answer[:_tail_fragment.start(1) + 1].rstrip()
    _tail_short_line = re.search(r"([.!?])\s*\n+\s*([^\n]{1,50}[.!?]?)\s*$", answer)
    if _tail_short_line and _tail_short_line.start(1) > 30:
        answer = answer[:_tail_short_line.start(1) + 1].rstrip()

    results.append({
        "idx": real_idx, "question": q, "document": doc,
        "right_answer": right, "hallucinated_answer_reference": halluc,
        "extraction": extraction, "answer": answer,
    })
    with open(f"/workspace/results_{LABEL}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"  {i+1}/{len(rows)} gegenereerd ({time.time()-_t0:.0f}s)", flush=True)

print(f"=== OPGESLAGEN: /workspace/results_{LABEL}.json ({len(results)} items) ===", flush=True)
print("KLAAR_MET_ALLES", flush=True)
