# finetune_qwen.py
import os

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_dataset
import torch

MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DATASET = "tazwarrrr/cuda-to-rocm-wavefront-bugs"
OUTPUT = "/workspace/rocmport-qwen-finetuned"
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Set HF_TOKEN in the environment before running fine-tuning.")
REPO_ID = "tazwarrrr/rocmport-qwen-wavefront-finetuned"

os.makedirs(OUTPUT, exist_ok=True)

# Load dataset
ds = load_dataset(DATASET)


def format_example(example):
    return {
        "text": f"""### Task: Fix CUDA code for AMD wavefront-64
### Bug Category: {example['bug_category']}
### Risk Level: {example['risk_level']}
### Kernel Type: {example['kernel_type']}
### Original CUDA (contains AMD bug):
{example['cuda_snippet']}
### hipify output (bug still present):
{example['hip_naive']}
### Why it fails on AMD gfx942:
{example['explanation']}
### Corrected HIP for AMD wavefront-64:
{example['hip_corrected']}"""
    }


formatted = ds.map(format_example)
if hasattr(formatted, "keys"):
    train_split = "train" if "train" in formatted else next(iter(formatted.keys()))
    train_dataset = formatted[train_split]
else:
    train_dataset = formatted

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
if torch.cuda.is_available():
    model.to("cuda")

# LoRA config
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training
args = TrainingArguments(
    output_dir=OUTPUT,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    learning_rate=2e-4,
    bf16=torch.cuda.is_available(),
    fp16=False,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=args
)

trainer.train()
trainer.save_model(OUTPUT)

# Push to HuggingFace
merged_model = model.merge_and_unload()
merged_model.push_to_hub(REPO_ID, token=HF_TOKEN)
tokenizer.push_to_hub(REPO_ID, token=HF_TOKEN)
print("Done. Model pushed to HuggingFace.")
