import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset

def train(data_path, model_name="microsoft/phi-2"):
    os.makedirs("/workspace/lattice/checkpoints", exist_ok=True)
    
    with open(data_path) as f:
        data = json.load(f)
    
    dataset = Dataset.from_list([{"text": f"### Instruction:\n{d['instruction']}\n\n### Input:\n{d['input']}\n\n### Response:\n{d['output']}"} for d in data])
    print(f"Loaded {len(dataset)} examples")
    
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","dense"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    
    tokenized = dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=2048, padding="max_length"), remove_columns=["text"])
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="/workspace/lattice/checkpoints", num_train_epochs=3, per_device_train_batch_size=4, save_steps=50, logging_steps=10, fp16=True, report_to="none"),
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    trainer.save_model("/workspace/lattice/final")
    tokenizer.save_pretrained("/workspace/lattice/final")
    print("✅ Saved to /workspace/lattice/final")

if __name__ == "__main__":
    import sys
    train(sys.argv[1] if len(sys.argv) > 1 else "/workspace/training_data_alpaca.json")
