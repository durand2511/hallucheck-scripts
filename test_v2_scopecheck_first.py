import os, json, time
_ns = {}
with open("/workspace/eval_extract_compose_gemma.py", encoding="utf-8") as _f:
    _all_lines = _f.readlines()
_cutoff = next(i for i, l in enumerate(_all_lines) if l.startswith('print(f"=== Extract-then-compose'))
exec("".join(_all_lines[:_cutoff]), _ns)
EXTRACT_SYSTEM = _ns["EXTRACT_SYSTEM"]
COMPOSE_SYSTEM_BASE = _ns["COMPOSE_SYSTEM"]

# Zet de scope-check als EERSTE instructie i.p.v. aan het einde toegevoegd -- de hypothese is
# dat het model zijn openingszet ("Manufactured Housing -> QUALIFIES") al vastlegt voordat het
# een regel aan het eind van een lange prompt bereikt. Door de check vooraan te zetten, moet het
# model die stap al zetten voordat het de feitenlijst induikt.
SCOPE_CHECK_FIRST = (
    "BEFORE doing anything else: check whether the facts list repeatedly names a specific, "
    "narrowly-scoped program (e.g. an acronym or named plan) as the source of every rule. If so, "
    "and the question itself does NOT use that specific program's name, your very first reasoning "
    "step must be to state this scope mismatch explicitly, before evaluating any individual fact. "
    "In that case your FINAL ANSWER must say you cannot confirm eligibility for the general term "
    "the question uses, since the document's rules are all scoped to that narrower named program "
    "only -- do not let an individual fact 'qualifying' override this scope check.\n\n"
)
COMPOSE_SYSTEM = SCOPE_CHECK_FIRST + COMPOSE_SYSTEM_BASE
print(f"=== Scope-check VOORAAN geplaatst: {len(SCOPE_CHECK_FIRST)} tekens (totaal nu {len(COMPOSE_SYSTEM)}) ===", flush=True)

import torch, re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL = "google/gemma-4-31B-it"
ADAPTER = "/workspace/dpo_gemma31b_grounding-adapter_v2"
IDX_LIST = [int(x) for x in os.environ.get("IDX_LIST", "199").split(",")]
LABEL = os.environ.get("LABEL", "v2_scopecheck_first")

data = load_dataset("google/FACTS-grounding-public", "examples")["public"]
rows = [data[i] for i in IDX_LIST]

BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": 0}, dtype=torch.bfloat16, quantization_config=BNB)
model = PeftModel.from_pretrained(base, ADAPTER)

_eot_id = tok.convert_tokens_to_ids("<end_of_turn>")
_stop_ids = {tok.eos_token_id}
if isinstance(_eot_id, int) and _eot_id >= 0:
    _stop_ids.add(_eot_id)

def gen(system, user, max_new, max_continuations=1):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    full_text = ""
    for _round in range(1 + max_continuations):
        _t = time.time()
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.15,
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

def clean(answer):
    _leak = re.search(r"\bthought\b|\bwait\b[*_]{0,2}[,.\-—]|\blet me restart\b", answer, flags=re.IGNORECASE)
    if _leak and _leak.start() > 30:
        answer = answer[:_leak.start()].rstrip()
    _degenerate = re.search(r"(.{2,80}?)\1{3,}", answer, flags=re.DOTALL)
    if _degenerate and _degenerate.start() > 30:
        answer = answer[:_degenerate.start()].rstrip()
    return answer

results = []
for i, row in enumerate(rows):
    q = row["user_request"]
    doc = row["context_document"][:6000]
    extract_user = f"QUESTION: {q}\n\nDOCUMENT:\n{doc}"
    extraction = gen(EXTRACT_SYSTEM, extract_user, 500)
    _end = re.search(r"</self_described>", extraction, flags=re.IGNORECASE)
    if _end:
        extraction = extraction[:_end.end()]
    compose_user = f"QUESTION: {q}\n\nFACTS LIST (extracted earlier):\n{extraction}"
    answer = clean(gen(COMPOSE_SYSTEM, compose_user, 600))
    results.append({"idx": IDX_LIST[i], "question": q, "extraction": extraction, "answer": answer})
    print(f"  {i+1}/{len(rows)} gegenereerd", flush=True)
    with open(f"/workspace/results_{LABEL}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

print("KLAAR_MET_ALLES", flush=True)
