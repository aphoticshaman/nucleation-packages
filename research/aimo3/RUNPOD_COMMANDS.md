# RunPod Quick Commands

## 1. KAGGLE CLI SETUP (run first)

```bash
pip install kaggle
mkdir -p ~/.kaggle

# Replace with YOUR credentials from kaggle.com -> Settings -> API -> Create New Token
cat > ~/.kaggle/kaggle.json << 'EOF'
{"username": "YOUR_USERNAME", "key": "YOUR_API_KEY"}
EOF
chmod 600 ~/.kaggle/kaggle.json

# Verify
kaggle datasets list --mine
```

## 2. UPLOAD WHEEL DATASET

```bash
# Create wheels
mkdir -p /workspace/ryanaimo-vllm-wheels
cd /workspace/ryanaimo-vllm-wheels

pip download --python-version 3.10 \
    --platform manylinux_2_17_x86_64 \
    --platform manylinux2014_x86_64 \
    --only-binary=:all: \
    vllm transformers accelerate safetensors msgspec \
    sentencepiece tokenizers huggingface_hub pynvml \
    ray aiohttp uvloop polars sympy scipy mpmath \
    outlines xformers grpcio protobuf -d . 2>&1 || true

pip download --python-version 3.10 \
    vllm transformers accelerate polars sympy msgspec -d . 2>&1 || true

# Create metadata (CHANGE USERNAME)
cat > dataset-metadata.json << 'EOF'
{
  "title": "ryanaimo-vllm-wheels",
  "id": "YOUR_USERNAME/ryanaimo-vllm-wheels",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

# Upload
kaggle datasets create -p . --dir-mode zip
```

## 3. DOWNLOAD + QUANTIZE DEEPSEEK (AWQ)

```bash
pip install huggingface_hub autoawq

# Download
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --local-dir /workspace/deepseek-r1-32b \
    --local-dir-use-symlinks False

# Quantize (Python)
python << 'PYEOF'
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained(
    "/workspace/deepseek-r1-32b",
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("/workspace/deepseek-r1-32b", trust_remote_code=True)

# Math-focused calibration
calib_data = [
    "Find all integers x such that x^2 - 5x + 6 = 0.",
    "Calculate the sum from k=1 to 100 of k^2.",
    "How many divisors does 2024 have?",
    "Let f(x) = x^3 - 6x^2 + 11x - 6. Find all roots.",
    "In how many ways can 10 be written as sum of positive integers?",
]

model.quantize(tokenizer, quant_config={
    "zero_point": True, "q_group_size": 128,
    "w_bit": 4, "version": "GEMM"
}, calib_data=calib_data)

model.save_quantized("/workspace/deepseek-r1-32b-awq")
tokenizer.save_pretrained("/workspace/deepseek-r1-32b-awq")
print("Done!")
PYEOF

# Upload to Kaggle (CHANGE USERNAME)
cd /workspace/deepseek-r1-32b-awq
cat > dataset-metadata.json << 'EOF'
{
  "title": "deepseek-r1-32b-awq",
  "id": "YOUR_USERNAME/deepseek-r1-32b-awq",
  "licenses": [{"name": "apache-2.0"}]
}
EOF
kaggle datasets create -p . --dir-mode zip
```

## 4. DOWNLOAD + QUANTIZE QWEN-CODER (AWQ)

```bash
# Download
huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct \
    --local-dir /workspace/qwen-coder-32b \
    --local-dir-use-symlinks False

# Quantize
python << 'PYEOF'
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained(
    "/workspace/qwen-coder-32b",
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("/workspace/qwen-coder-32b", trust_remote_code=True)

# Code-focused calibration
calib_data = [
    "Write Python code to find all prime numbers up to 1000.",
    "Implement a function to compute the nth Fibonacci number.",
    "Write code to solve a system of linear equations using numpy.",
    "Create a recursive function to compute factorials.",
    "Write Python to enumerate all permutations of a list.",
]

model.quantize(tokenizer, quant_config={
    "zero_point": True, "q_group_size": 128,
    "w_bit": 4, "version": "GEMM"
}, calib_data=calib_data)

model.save_quantized("/workspace/qwen-coder-32b-awq")
tokenizer.save_pretrained("/workspace/qwen-coder-32b-awq")
print("Done!")
PYEOF

# Upload to Kaggle (CHANGE USERNAME)
cd /workspace/qwen-coder-32b-awq
cat > dataset-metadata.json << 'EOF'
{
  "title": "qwen-coder-32b-awq",
  "id": "YOUR_USERNAME/qwen-coder-32b-awq",
  "licenses": [{"name": "apache-2.0"}]
}
EOF
kaggle datasets create -p . --dir-mode zip
```

## 5. FINE-TUNE ON AIME/AMC DATA (OPTIONAL - needs A100)

```bash
pip install trl peft bitsandbytes datasets

python << 'PYEOF'
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import torch

# Load base model in 4-bit for training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "/workspace/deepseek-r1-32b",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("/workspace/deepseek-r1-32b")
tokenizer.pad_token = tokenizer.eos_token

# LoRA config for efficient fine-tuning
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Load AIME dataset (you'll need to prepare this)
# Format: {"prompt": "problem text", "completion": "solution with \\boxed{answer}"}
dataset = load_dataset("json", data_files="/workspace/aime_train.jsonl")

# Training config
training_args = SFTConfig(
    output_dir="/workspace/deepseek-r1-32b-aime-lora",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args,
    tokenizer=tokenizer,
    max_seq_length=4096,
)

trainer.train()
trainer.save_model()
print("Fine-tuning complete!")
PYEOF
```

## FINAL: Your Kaggle Datasets

After running, you'll have these on Kaggle:
1. `YOUR_USERNAME/ryanaimo-vllm-wheels` - All dependencies
2. `YOUR_USERNAME/deepseek-r1-32b-awq` - Reasoning model (AWQ 4-bit)
3. `YOUR_USERNAME/qwen-coder-32b-awq` - Code execution model (AWQ 4-bit)

Attach all three to your notebook!
