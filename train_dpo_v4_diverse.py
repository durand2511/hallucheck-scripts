#!/usr/bin/env python3
# V4: gerichte boosterronde met DIVERSE, geparafraseerde trainingsvoorbeelden (2-3 varianten per
# hardnekkig patroon: idx199/380/723/806) i.p.v. de v3-poging die maar 1 exacte zin-paar per
# patroon gebruikte en niet generaliseerde (zeer lage trainingsloss maar geen gedragsverandering
# bij greedy decoding -- DPO optimaliseert relatieve voorkeur tussen 2 vaste teksten, niet
# noodzakelijk de argmax-generatietrajectorie). Vertrekt vanaf v2 (4/8 al correct), niet vanaf v3.
import os, json, torch, gc
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import DPOTrainer, DPOConfig

MODEL = "google/gemma-4-31B-it"
BASE_ADAPTER = "/workspace/dpo_gemma31b_grounding-adapter_v2"
DATA = "/workspace/train_dpo_v4_diverse.jsonl"
OUT = "/workspace/dpo_gemma31b_grounding-adapter_v4"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
print(f"DPO-paren (diverse booster): {len(rows)} -- idx {[r['idx'] for r in rows]}", flush=True)

def render_prompt(messages):
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

examples = []
for r in rows:
    examples.append({
        "prompt": render_prompt(r["prompt"]),
        "chosen": r["chosen"],
        "rejected": r["rejected"],
    })
ds = Dataset.from_list(examples)

BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
print("=== Basismodel laden (4bit) + v2-adapter als trainable ===", flush=True)
base = AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": 0}, dtype=torch.bfloat16, quantization_config=BNB)
model = PeftModel.from_pretrained(base, BASE_ADAPTER, is_trainable=True)
model.config.use_cache = False
model.print_trainable_parameters()

MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))
EPOCHS = int(os.environ.get("EPOCHS", "4"))

cfg_kwargs = dict(
    output_dir=OUT,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_steps=3,
    logging_steps=1,
    save_strategy="no",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    beta=0.1,
    use_liger_kernel=True,
    max_length=8192,
    report_to=[],
)
if MAX_STEPS > 0:
    cfg_kwargs["max_steps"] = MAX_STEPS
    print(f"=== DRY-RUN MODUS: max_steps={MAX_STEPS} ===", flush=True)
else:
    cfg_kwargs["num_train_epochs"] = EPOCHS
    print(f"=== V4: {EPOCHS} epochs op {len(rows)} diverse voorbeelden = {EPOCHS*len(rows)} stappen ===", flush=True)
cfg = DPOConfig(**cfg_kwargs)

trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
print("=== TRAINING START ===", flush=True)
trainer.train()
print("=== TRAINING KLAAR, opslaan naar", OUT, "===", flush=True)
trainer.save_model(OUT)
tok.save_pretrained(OUT)
print("DPO-TRAINING_KLAAR_V4", flush=True)
