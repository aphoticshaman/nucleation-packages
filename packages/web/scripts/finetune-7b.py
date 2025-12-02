#!/usr/bin/env python3
"""
LatticeAI 7B Fine-Tuning Script
Run on RunPod with A5000 (24GB) or A100 (40GB)

Usage:
  pip install transformers peft datasets accelerate bitsandbytes
  python finetune-7b.py --data training_data_alpaca.json

Cost: ~$1-5 for 1K-10K examples on A5000
Time: 2-12 hours depending on dataset size
"""

import argparse
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_data(data_path: str) -> Dataset:
    """Load training data in Alpaca format."""
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Convert to prompt format
    formatted = []
    for item in data:
        text = f"""### Instruction:
{item['instruction']}

### Input:
{item['input']}

### Response:
{item['output']}"""
        formatted.append({"text": text})

    return Dataset.from_list(formatted)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to training data JSON")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B", help="Base model")
    parser.add_argument("--output", default="./lattice-lora", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    dataset = load_data(args.data)
    print(f"Loaded {len(dataset)} examples")

    # Quantization config for memory efficiency
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit quantization")
    elif args.use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        print("Using 8-bit quantization")

    print(f"Loading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not bnb_config else None,
    )

    if bnb_config:
        model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize dataset
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=2048,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(tokenize, remove_columns=["text"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit" if bnb_config else "adamw_torch",
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {args.output}...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    print("\n=== Training Complete ===")
    print(f"Model saved to: {args.output}")
    print(f"To use: Load with `PeftModel.from_pretrained('{args.output}')`")

if __name__ == "__main__":
    main()
