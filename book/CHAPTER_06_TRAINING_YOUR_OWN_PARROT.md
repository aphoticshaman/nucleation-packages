# Chapter 6: Training Your Own Parrot

You've learned to use parrots. You've learned to prompt them effectively. You've learned to build systems around them and catch their lies.

Now let's grow our own.

This chapter is where we cross from consumer to producer—from using other people's models to creating our own. By the end, you'll understand how to fine-tune existing models for your specific needs, deploy them for your own use, and potentially save hundreds of dollars per month while gaining capabilities the big providers don't offer.

This is also my story. I went from competing in a Kaggle challenge to training and deploying 70-billion-parameter models on HuggingFace. The journey from "user of AI" to "builder of AI" is more accessible than you think.

Let's do it.

---

## Why Train Your Own?

First, the obvious question: why bother?

OpenAI, Anthropic, and Google offer powerful models through simple APIs. You type, you pay, you get results. Why would anyone want to run their own models?

### Reason 1: Cost

API calls add up fast.

A heavy Claude user might spend $200-500/month on API access. A business integrating GPT-4 into their product might spend thousands. At scale, API costs become a major line item.

Running your own model has upfront costs (compute for training, hosting for inference) but marginal costs approach zero. Once you've fine-tuned a model and deployed it, additional queries are nearly free.

**Break-even example:**
- API cost: $200/month
- Self-hosted cost: $50/month (cloud GPU)
- Savings: $150/month, $1,800/year

For individuals, this means independence. For businesses, this means margins.

### Reason 2: Customization

OpenAI's models are generalists. They're trained on everything, optimized for no one.

Your model can be a specialist. Fine-tuned on your data, optimized for your use case, speaking in your voice.

**Examples:**
- A legal model trained on your firm's documents and style
- A coding model specialized in your tech stack
- A support model that knows your product inside out
- A writing model that matches your specific tone

Generic models are okay at everything. Specialized models are great at their thing.

### Reason 3: Privacy

API calls send your data to someone else's servers.

For sensitive applications—medical, legal, financial, personal—this might be unacceptable. Self-hosted models keep data on your infrastructure.

### Reason 4: Control

APIs can change without notice. Pricing changes. Capability changes. Rate limits. Terms of service.

Your own model is yours. No one can deprecate it, change its behavior, or price you out.

### Reason 5: Learning

The deepest understanding comes from doing.

Reading about how models work is one thing. Actually training one—watching loss curves, debugging failures, seeing what data choices matter—is another level entirely.

If you want to truly understand these systems, train one.

---

## The Training Landscape

Before diving into how, let's understand what options exist.

### Option 1: Train from Scratch

Start with random weights. Train on trillions of tokens. End up with GPT-4.

**Realistic?** No. This costs millions of dollars and requires infrastructure most organizations don't have.

**When it makes sense:** You're a major AI lab or a well-funded research institution.

### Option 2: Pre-train on Custom Data

Take an existing architecture, train from scratch on your specific data.

**Realistic?** Maybe. Requires substantial compute but could be worthwhile for highly specialized domains.

**When it makes sense:** You have massive domain-specific data and need a model that's never seen general text.

### Option 3: Fine-tune an Existing Model

Take a pre-trained model (LLaMA, Mistral, etc.), adjust its weights using your data.

**Realistic?** Yes! This is the sweet spot. Pre-training teaches the model language; fine-tuning teaches it your task.

**When it makes sense:** Almost always the right starting point.

### Option 4: Use Efficient Fine-Tuning (LoRA, QLoRA)

Fine-tune only a small fraction of the model's weights. Keep most of the model frozen.

**Realistic?** Very yes. This can be done on consumer hardware for smaller models, or modest cloud compute for larger ones.

**When it makes sense:** You want the benefits of fine-tuning without the compute costs. This is usually the best choice.

We'll focus on Options 3 and 4—fine-tuning and efficient fine-tuning. This is where individuals and small teams can actually operate.

---

## Understanding Fine-Tuning

Fine-tuning is transfer learning for language models.

**Pre-training:** The model learns general language patterns from massive data
**Fine-tuning:** The model learns specific patterns from your data

Think of it like education:
- Pre-training = General education (K-12)
- Fine-tuning = Specialized training (medical school, law school, trade school)

A doctor learns to read before learning medicine. A model learns language before learning your domain.

### What Fine-Tuning Changes

When you fine-tune, you're adjusting the model's weights based on your data.

**Full fine-tuning:** Adjust all parameters (expensive, powerful)
**Partial fine-tuning:** Freeze early layers, adjust later layers
**Adapter-based fine-tuning:** Add small trainable modules, keep main model frozen

Modern best practice: **LoRA (Low-Rank Adaptation)** and its variants.

### How LoRA Works

The key insight: you don't need to modify all parameters.

Instead of updating a weight matrix W directly:
```
W_new = W_old + ΔW
```

LoRA approximates the change with a low-rank decomposition:
```
ΔW ≈ A × B
where A is (d × r) and B is (r × d)
r << d (r is the "rank", typically 8-64)
```

This reduces trainable parameters by 1000x or more while maintaining most of the fine-tuning benefit.

**Example:**
- Full fine-tuning 7B model: ~7 billion trainable parameters
- LoRA fine-tuning 7B model: ~8 million trainable parameters
- Memory required: ~10% of full fine-tuning

LoRA makes fine-tuning accessible on consumer GPUs.

### QLoRA: Even More Efficient

QLoRA combines LoRA with quantization:
- Quantize the base model to 4-bit (reduces memory 4x)
- Train LoRA adapters on top
- Result: Fine-tune 70B models on a single GPU

This is what unlocks training genuinely large models without enterprise hardware.

---

## The Fine-Tuning Process

Here's how fine-tuning actually works, step by step:

### Step 1: Choose Your Base Model

Options include:
- **LLaMA 3 family:** Meta's open models, excellent quality
- **Mistral:** Efficient, strong performance
- **Qwen:** Alibaba's models, good multilingual support
- **Phi:** Microsoft's small but capable models
- **Gemma:** Google's open models

For most use cases, start with a 7B-13B model. These balance capability with trainability. If you need more power and have the compute, 70B models are accessible via QLoRA.

### Step 2: Prepare Your Data

Fine-tuning data should be:
- **Formatted correctly:** Most models expect a specific format (conversations, instructions, etc.)
- **High quality:** Garbage in, garbage out. Curate carefully.
- **Representative:** Should cover the range of tasks you want the model to handle
- **Appropriately sized:** More isn't always better; 1,000 high-quality examples often beats 100,000 low-quality ones

**Common formats:**

Instruction format:
```json
{
  "instruction": "What is the capital of France?",
  "response": "The capital of France is Paris."
}
```

Conversation format:
```json
{
  "conversations": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

### Step 3: Set Up Training Environment

Tools you'll need:
- **Python** with PyTorch
- **Transformers library** (Hugging Face)
- **PEFT library** (for LoRA)
- **BitsAndBytes** (for quantization)
- **Axolotl or TRL** (training frameworks)

If you don't have a local GPU:
- **Google Colab:** Free tier has limited GPUs, Pro is ~$10/month
- **RunPod/Vast.ai:** GPU rentals, pay-per-hour
- **Lambda Labs:** Higher-end GPUs, professional setup
- **Cloud providers:** AWS, GCP, Azure all offer GPU instances

### Step 4: Configure Training

Key hyperparameters:
- **Learning rate:** Typically 1e-4 to 5e-5 for fine-tuning
- **Batch size:** As large as memory allows (gradient accumulation helps)
- **Epochs:** 1-3 usually sufficient; more risks overfitting
- **LoRA rank (r):** 8-64, higher = more parameters = more capacity
- **LoRA alpha:** Usually 2× the rank
- **Target modules:** Which attention modules to adapt (typically q, k, v, o projections)

**Example Axolotl config:**
```yaml
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
load_in_4bit: true
adapter: qlora
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
learning_rate: 2e-4
num_epochs: 2
micro_batch_size: 2
gradient_accumulation_steps: 8
```

### Step 5: Train

Start training and watch:
- **Loss curve:** Should decrease smoothly
- **Memory usage:** Watch for OOM errors
- **Checkpoints:** Save periodically in case of crashes

Training time varies:
- Small model, small data: Hours
- Large model, large data: Days
- 70B model: Many days even with QLoRA

### Step 6: Evaluate

After training:
- Run on held-out test set
- Check for overfitting (test loss much higher than train loss)
- Manual evaluation: Does it do what you want?
- Compare to base model: Is it actually better?

### Step 7: Merge and Deploy

For LoRA models:
- Merge adapter weights into base model (optional but simplifies deployment)
- Export to format for inference (GGUF for local, safetensors for HuggingFace)
- Deploy on your chosen platform

---

## Deployment Options

You've trained your model. Now what?

### Local Deployment

Run on your own hardware.

**Tools:**
- **Ollama:** Simple local deployment, handles everything
- **llama.cpp:** Lower-level, more control
- **vLLM:** High-performance inference server
- **Text Generation WebUI:** Nice GUI for local models

**Pros:** Total control, no ongoing costs, maximum privacy
**Cons:** Limited by your hardware, need to handle reliability

### Cloud Deployment

Run on rented infrastructure.

**Options:**
- **HuggingFace Inference Endpoints:** Easy deployment from HuggingFace repos
- **Modal:** Serverless GPU inference
- **Replicate:** Run models as API endpoints
- **RunPod Serverless:** Pay-per-request GPU inference

**Pros:** Scale as needed, no hardware management
**Cons:** Ongoing costs, data leaves your control

### HuggingFace Model Hub

Share your model publicly (or privately).

**Why do this:**
- Portfolio/credibility
- Community feedback
- Others can use/improve your work
- Free hosting for model weights

**My models on HuggingFace:** [huggingface.co/aphoticshaman](https://huggingface.co/aphoticshaman)

---

## My Journey: From AIMO3 to Deployed Models

Let me tell you how I got here.

It started with a Kaggle competition—the AI Mathematical Olympiad (AIMO). The prize: solve IMO-level math problems with AI. The approach most people were using: majority voting on samples from large models.

I was watching these models produce answers and noticing something strange: when they got math wrong, they weren't randomly wrong. They were *consistently* wrong in specific ways. Near-misses clustered. The errors had structure.

That observation became the CIC framework (Part III).

But to test these ideas, I needed models. Lots of models. API calls were expensive. I needed to run inference myself.

So I learned to fine-tune. Started with 7B models. Learned the tooling, the configurations, the failure modes. Graduated to 13B. Then 70B with QLoRA.

Now I have quantized 70B models running on HuggingFace, specialized for mathematical reasoning. The same models I use for research. The infrastructure costs me a fraction of what API access would—and I have complete control.

The path:
1. User → frustrated with API costs and limitations
2. Experimenter → tried fine-tuning tutorials
3. Practitioner → trained models that actually worked
4. Builder → deployed models, developed frameworks
5. Researcher → discovered CIC, wrote this book

You're on this path now. How far you go depends on what you need.

---

## When Fine-Tuning Makes Sense

Fine-tuning isn't always the answer. Here's when it is:

### Good Use Cases

**Specific domain expertise:** You have data in a domain where general models struggle
**Particular style/voice:** You need the model to write in a specific way consistently
**Cost reduction:** You're spending too much on API calls
**Privacy requirements:** You can't send data to external APIs
**Research and understanding:** You want to learn how these systems work

### Bad Use Cases

**General chat:** General models already do this well
**Simple tasks:** Prompting usually solves it cheaper
**Frequently changing needs:** Retraining is slower than updating prompts
**Limited data:** Less than 100 examples probably won't help much
**Limited compute:** If you can't run the model, you can't use it

### The Decision Framework

1. Can prompting solve this? If yes → Don't fine-tune
2. Do you have good data? If no → Don't fine-tune (yet)
3. Is cost/privacy/control a priority? If yes → Consider fine-tuning
4. Do you have compute (or budget for it)? If no → Use APIs for now
5. All yes → Fine-tune!

---

## Practical Exercise: Your First Fine-Tune

Let's actually do this.

### Objective

Fine-tune a small model to respond in a specific style.

### Materials

- Google Colab (free tier or Pro)
- HuggingFace account
- 50-100 examples of desired input/output

### Process

**Step 1:** Prepare 50 examples of conversations in the style you want:
```json
{"conversations": [
  {"role": "user", "content": "Your input..."},
  {"role": "assistant", "content": "Desired output style..."}
]}
```

**Step 2:** Upload to Google Drive or HuggingFace dataset

**Step 3:** Open Colab notebook (I'll provide link in repo)

**Step 4:** Run the training cells (takes 1-2 hours on free Colab)

**Step 5:** Test your fine-tuned model

**Step 6:** If it works, upload to HuggingFace

### What You'll Learn

- The actual process (less mysterious than it seems)
- What can go wrong (and how to fix it)
- Whether fine-tuning is worth it for your use case

---

## Looking Ahead

You now have the practical skills to:
- Use LLMs effectively (Chapters 1-2)
- Build systems around them (Chapters 3-4)
- Catch their lies (Chapter 5)
- Train your own (Chapter 6)

That's Part I complete. You're a power user—more capable than 95% of people using these tools.

Part II goes deeper: how attention actually works, what networks learn, why training dynamics produce the behaviors we observe. This is where the math starts.

Part III introduces my research: the CIC framework for reliable inference. This is where we solve the fundamental problems identified in Chapter 5.

Part IV deploys that research across 50 innovations for real systems.

Part V looks ahead: safety, alignment, where this is all going.

The parrot metaphor has served us well. But in Part II, we're going to dissect the parrot—see what's actually inside. The metaphor will stretch. That's because reality is stranger than the metaphor.

Let's go see what's inside.

---

*Chapter 6 Summary:*

- Training your own model: cost savings, customization, privacy, control, learning
- LoRA/QLoRA: efficient fine-tuning, accessible on consumer hardware
- The process: base model → data prep → train → evaluate → deploy
- Deployment options: local, cloud, HuggingFace
- My journey: AIMO3 competition → CIC research → deployed 70B models
- When to fine-tune vs. when to just prompt

*Part I complete. You're now a power user. Part II: understanding how these systems actually work.*
