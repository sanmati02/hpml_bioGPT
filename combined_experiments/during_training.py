"""
during_training.py

Runs BioGPT-Large-PubMedQA using:
- Structured Pruning (L2 on neurons)
- QAT with float16
- torch.compile with AOT Eager or InductorMax Autotune backend
Logs inference metrics using Weights & Biases.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AdamW
from datasets import Dataset
import json
import wandb
import pandas as pd
from tqdm import tqdm
import re
import time
import subprocess
import argparse
from sklearn.metrics import accuracy_score

# Parse command-line argument for pruning strategyparser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument("--backend", type=str, choices=["aot_eager", "inductor"], default="aot_eager")
args = parser.parse_args()

# Load and preprocess PubMedQA dev set
with open("../evaluation/dev_set.json") as f:
    raw_data = json.load(f)

# Extract relevant fields from raw JSON into structured examples
examples = []
for pmid, item in raw_data.items():
    examples.append({
        "PMID": pmid,
        "question": item["QUESTION"],
        "context": " ".join(item["CONTEXTS"]),
        "label": item["final_decision"]
    })

# Load model tokenizer & Dataset (dev)
train_data = Dataset.from_list(examples)
tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large-PubMEDQA")

# Preprocessing for causal LM training — prompt appended with label, loss masked on prompt
def preprocess(example):
    prompt = example["question"] + " Context: " + example["context"] + ". Answer in yes or no:"
    full_text = prompt + " " + example["label"]
    enc = tokenizer(full_text, padding="max_length", truncation=True, max_length=256)
    prompt_ids = tokenizer(prompt, padding="max_length", truncation=True, max_length=256)["input_ids"]
    enc["labels"] = [-100 if i < sum(p != tokenizer.pad_token_id for p in prompt_ids) else t for i, t in enumerate(enc["input_ids"])]
    return enc

train_data = train_data.map(preprocess)

# Collate function for batched token tensors
def collate_fn(batch):
    tensor_keys = ["input_ids", "attention_mask", "labels"]
    return {k: torch.tensor([f[k] for f in batch]) for k in tensor_keys if k in batch[0]}

loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn)

# Load model and tokenizer and default config and optimizer
config = AutoConfig.from_pretrained("microsoft/BioGPT-Large-PubMEDQA")
model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large-PubMEDQA", config=config)
model.gradient_checkpointing_enable()
model.to("cuda")

# Apply pruning strategy of structured to Linear layers
def apply_structured_pruning(model, amount=0.2):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)

apply_structured_pruning(model)

# torch.compile arguments 
compile_kwargs = {"backend": args.backend}
if args.backend == "inductor":
    compile_kwargs["mode"] = "max-autotune"
model = torch.compile(model, **compile_kwargs)


optimizer = AdamW(model.parameters(), lr=1e-5)

# AMP setup (Automatic Mixed Precision)
scaler = torch.amp.GradScaler()

# Fine-tuning model with pruning strategy applied on model - for 5 epochs 
for epoch in range(5):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to("cuda") for k, v in batch.items()}
        optimizer.zero_grad()
        # Training with float16 AMP
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
             # Mixed precision context
            out = model(**batch, use_cache=False)
            loss = out.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        print(f"Epoch {epoch+1} batch loss: {loss.item():.4f}")
    print(f"\U0001f4c9 Epoch {epoch+1} avg loss: {total_loss/len(loader):.4f}")

# Remove pruning wrappers to finalize model
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and hasattr(module, "weight_orig"):
        prune.remove(module, "weight")

# # Load test data
with open("../evaluation/test_set.json") as f:
    test_data = json.load(f)
with open("../evaluation/test_ground_truth.json") as f:
    test_gt = json.load(f)


# Join QA with labels
# Clean and prepare questions and ground-truth labels
questions = [(v["QUESTION"].strip() + "." if not v["QUESTION"].strip().endswith(".") else v["QUESTION"].strip()) for v in test_data.values()]
y_true = list(test_gt.values())

# Queries GPU statistics including utilization, memory, power, and temperature.
def get_gpu_info():
    result = subprocess.run([
        'nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,power.draw,temperature.gpu',
        '--format=csv,noheader,nounits'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    vals = result.stdout.decode("utf-8").strip().split(",")
    return {
        "gpu_utilization": float(vals[0]),
        "memory_utilization": float(vals[1]),
        "power_usage": float(vals[2]),
        "temperature": float(vals[3])
    }

# Initialize wandb
wandb.init(project="biogpt-pubmedqa", name=f"qat+structured+fp16+{args.backend}", config={
    "pruning": "structured", "precision": "float16", "compile_backend": args.backend, "task": "QA-PubMedQA"
})

# Inference on test set
latencies, answers = [], []
gpu_utilization_list, memory_utilization_list, power_usage_list, temperature_list = [], [], [], []

model.eval()
for q in tqdm(questions):
    q += " Answer in the following format in yes or no. the answer to the question given the context is"
    inputs = tokenizer(q, return_tensors="pt").to("cuda")
     # Inference timing
    start_time = time.time()
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        output = model.generate(**inputs, max_new_tokens=256, num_beams=1)
    latency = time.time() - start_time
    latencies.append(latency)

    # Log GPU stats
    gpu_metrics = get_gpu_info()
    gpu_utilization_list.append(gpu_metrics["gpu_utilization"])
    memory_utilization_list.append(gpu_metrics["memory_utilization"])
    power_usage_list.append(gpu_metrics["power_usage"])
    temperature_list.append(gpu_metrics["temperature"])
    wandb.log({**gpu_metrics, "latency": latency})

    # Decode answer
    # Parse model output to yes/no/maybe
    # Extract a clear yes/no/maybe answer from the decoded model output
    answers.append(tokenizer.decode(output[0], skip_special_tokens=True))

def convert(s):
    match = re.search(r"the answer to the question given the context is (yes|no|maybe)", s, re.I)
    return match.group(1).strip() if match else "failed"

hypothesis = [convert(a) for a in answers]


# Compute final metrics
accuracy = accuracy_score(y_true[:len(hypothesis)], hypothesis)
throughput = len(hypothesis) / sum(latencies)

# Log summary metrics to wandb
wandb.log({
    "accuracy": accuracy,
    "throughput": throughput,
    "avg_latency": sum(latencies)/len(latencies),
    "avg_gpu_utilization": sum(gpu_utilization_list)/len(gpu_utilization_list),
    "avg_memory_utilization": sum(memory_utilization_list)/len(memory_utilization_list),
    "avg_power_usage": sum(power_usage_list)/len(power_usage_list),
    "avg_temperature": sum(temperature_list)/len(temperature_list)
})
wandb.finish()
print(f"Done. Accuracy: {accuracy:.4f}, Throughput: {throughput:.2f}/s")

