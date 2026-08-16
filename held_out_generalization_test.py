import os, json, time
_ns = {}
with open("/workspace/eval_extract_compose_gemma.py", encoding="utf-8") as _f:
    _all_lines = _f.readlines()
_cutoff = next(i for i, l in enumerate(_all_lines) if l.startswith('print(f"=== Extract-then-compose'))
exec("".join(_all_lines[:_cutoff]), _ns)
EXTRACT_SYSTEM = _ns["EXTRACT_SYSTEM"]
COMPOSE_SYSTEM = _ns["COMPOSE_SYSTEM"]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL = os.environ.get("MODEL", "google/gemma-4-31B-it")
ADAPTER = os.environ.get("ADAPTER_PATH", "/workspace/dpo_gemma31b_grounding-adapter")
LABEL = os.environ.get("LABEL", "held_out")
BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")

CASES = [
    {
        "id": "held_out_A_most_specific_category",
        "question": (
            "I joined the gym during their opening month last year and got the special founding rate. "
            "It's now been 8 months. I want to cancel my membership. Since I've had a membership for way "
            "more than 30 days, can I cancel penalty-free by giving 30 days notice?"
        ),
        "document": (
            "CANCELLATION POLICY\n"
            "Standard Members may cancel their membership at any time with 30 days written notice, no "
            "early-termination fee applies.\n"
            "Founding Members (individuals who joined during the club's opening month and locked in the "
            "founding rate) are enrolled in a 24-month minimum-commitment agreement; cancelling before "
            "completing 24 months incurs an early-termination fee equal to 50% of the remaining months' "
            "dues. Founding Members become eligible for penalty-free cancellation only after 24 months of "
            "continuous membership."
        ),
        "expected": (
            "NO -- the person is specifically a Founding Member (joined during opening month, locked in "
            "founding rate), so the more specific Founding Member rule applies (24-month minimum "
            "commitment), not the generic Standard Member 30-day rule. At 8 months they have not completed "
            "24 months, so cancelling now triggers the early-termination fee, not a penalty-free cancellation."
        ),
    },
    {
        "id": "held_out_B_document_scope_match",
        "question": (
            "My coworker said hotel stays are an eligible purchase category according to company policy. "
            "I'm planning a personal vacation next month and want to book a hotel. Can I use my regular "
            "company credit card for personal hotel bookings under this policy?"
        ),
        "document": (
            "CORPORATE TRAVEL CARD PROGRAM -- ELIGIBLE PURCHASE CATEGORIES\n"
            "This document describes the purchase categories eligible for reimbursement under the Corporate "
            "Travel Card Program, issued exclusively to employees who travel for business at least twice per "
            "quarter.\n"
            "Eligible categories: airfare, hotel lodging, ground transportation (taxis, rideshare, rental "
            "cars), and business meals during travel.\n"
            "Restaurant meals, groceries, and hotel lodging are all eligible purchase categories under this "
            "program."
        ),
        "expected": (
            "NO -- this document only describes eligible categories for the Corporate Travel Card Program "
            "(business travel, and only for employees who travel at least twice/quarter), not for a 'regular "
            "company credit card' or for personal/vacation use. Hotel lodging being listed does not make it "
            "eligible for a personal vacation booking on a different card."
        ),
    },
]

print(f"=== Held-out generalisatietest ({len(CASES)} nieuwe, nooit-geziene gevallen), adapter={ADAPTER} ===", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": 0}, dtype=torch.bfloat16, quantization_config=BNB)
model = PeftModel.from_pretrained(base, ADAPTER) if ADAPTER else base

_eot_id = tok.convert_tokens_to_ids("<end_of_turn>")
_stop_ids = {tok.eos_token_id}
if isinstance(_eot_id, int) and _eot_id >= 0:
    _stop_ids.add(_eot_id)

def gen(system, user, max_new, max_continuations=1):
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
for c in CASES:
    extract_user = f"QUESTION: {c['question']}\n\nDOCUMENT:\n{c['document']}"
    extraction = gen(EXTRACT_SYSTEM, extract_user, 500)
    compose_user = f"QUESTION: {c['question']}\n\nFACTS LIST (extracted earlier):\n{extraction}"
    answer = gen(COMPOSE_SYSTEM, compose_user, 600)
    results.append({"id": c["id"], "question": c["question"], "expected": c["expected"],
                     "extraction": extraction, "answer": answer})
    print(f"  {c['id']} klaar", flush=True)
    with open(f"/workspace/results_{LABEL}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

print("KLAAR_MET_ALLES", flush=True)
