#!/usr/bin/env python3
"""
Combined finance + global intel training on Phi-2
Run on second pod parallel to grok_data training
"""
import json
import sys
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def train(finance_path, intel_path):
    # Load and merge datasets
    with open(finance_path) as f:
        finance = json.load(f)
    with open(intel_path) as f:
        intel = json.load(f)

    # Sample finance to balance (don't overwhelm with 54k)
    import random
    random.seed(42)
    finance_sample = random.sample(finance, min(10000, len(finance)))

    all_data = finance_sample + intel
    random.shuffle(all_data)
    print(f"Training on {len(all_data)} examples ({len(finance_sample)} finance + {len(intel)} intel)")

    # Format for training
    formatted = []
    for x in all_data:
        inst = x.get('instruction', '')
        inp = x.get('input', '')
        out = x.get('output', '')
        if inst and out:
            text = f"### Instruction:\n{inst}\n\n"
            if inp:
                text += f"### Input:\n{inp}\n\n"
            text += f"### Response:\n{out}"
            formatted.append({"text": text})

    dataset = Dataset.from_list(formatted)
    print(f"Formatted {len(dataset)} examples")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "dense"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize
    def tokenize(example):
        result = tokenizer(example["text"], truncation=True, max_length=1024, padding="max_length")
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized = dataset.map(tokenize, remove_columns=["text"])

    # Train
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/workspace/lattice-finance-intel",
            num_train_epochs=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=50,
            save_steps=500,
            save_total_limit=3,
            remove_unused_columns=False,
            warmup_steps=100,
        ),
        train_dataset=tokenized,
    )

    trainer.train()
    trainer.save_model("/workspace/lattice-finance-intel")
    tokenizer.save_pretrained("/workspace/lattice-finance-intel")
    print("Done! Saved to /workspace/lattice-finance-intel")

if __name__ == "__main__":
    train(
        sys.argv[1] if len(sys.argv) > 1 else "/workspace/finance_training_data.json",
        sys.argv[2] if len(sys.argv) > 2 else "/workspace/global_intel_training.json"
    )
