#!/usr/bin/env python3
"""
LatticeForge LoRA Training Script - Optimized for H200 (141GB VRAM)
Run: python train_h200.py
"""

import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Load training data
print("Loading training data...")
with open("/workspace/combined_intel_training.json", "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} examples")

# Format for training
def format_example(ex):
    inp = ex.get("input", "")
    if inp:
        return f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{inp}\n\n### Response:\n{ex['output']}"
    return f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"

formatted = [{"text": format_example(ex)} for ex in data]
dataset = Dataset.from_list(formatted)

print("Loading model...")
model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# LoRA config - larger rank for H200
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "dense"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training args optimized for H200
training_args = TrainingArguments(
    output_dir="/workspace/latticeforge-phi2-lora",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=50,
    save_steps=500,
    save_total_limit=3,
    warmup_ratio=0.03,
    optim="adamw_torch",
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    processing_class=tokenizer,
    max_seq_length=2048
)

print("Starting training...")
trainer.train()

print("Saving model...")
trainer.save_model("/workspace/latticeforge-phi2-lora-final")
tokenizer.save_pretrained("/workspace/latticeforge-phi2-lora-final")

print("Done! Model saved to /workspace/latticeforge-phi2-lora-final")
