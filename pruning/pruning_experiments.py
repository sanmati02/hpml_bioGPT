"""
pruning_experiments.py

Runs BioGPT-Large-PubMedQA for Pruning strategies (unstructured, structured, block) and logs inference metrics such as accuracy, latency, GPU stats using Weights & Biases.
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AdamW
from datasets import Dataset
import json
import wandb
import pandas as pd
from tqdm import tqdm
import re
import time
import subprocess
from sklearn.metrics import accuracy_score
import argparse

# Parse command-line argument for pruning strategyparser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=str, choices=["unstructured", "structured", "block"], required=True)
args = parser.parse_args()
strategy = args.strategy

# Load and preprocess PubMedQA dev set
with open("../../pubmedqa/data/pqal_fold0/dev_set.json") as f:
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
config = AutoConfig.from_pretrained("microsoft/BioGPT-Large-PubMedQA")
model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large-PubMEDQA", config=config)
model.to("cuda")
optimizer = AdamW(model.parameters(), lr=1e-5)

# Apply pruning strategy to Linear layers
def apply_pruning(model, strategy, amount=0.2):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if strategy == "unstructured":
                # Remove weights with smallest L1-norm, without regard to position
                prune.l1_unstructured(module, name="weight", amount=amount)
            elif strategy == "structured":
                 # Remove entire rows (dim=0) from the weight matrix based on L2-norm
                # Typically zeros out entire neurons
                prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            elif strategy == "block":
                # Randomly zero out blocks (columns in dim=1) for structured block sparsity
                # Useful for hardware-friendly sparsity
                prune.random_structured(module, name="weight", amount=amount, dim=1)

apply_pruning(model, strategy)

# Initialize wandb
wandb.init(project="biogpt-pubmedqa", name=f"{strategy}-pruning", config={"strategy": strategy})


# Fine-tuning model with pruning strategy applied on model - for 5 epochs 
for epoch in range(5):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to("cuda") for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(**batch, use_cache=False)
        loss = out.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
    print(f"Epoch {epoch+1} avg loss: {total_loss/len(loader):.4f}")

# Remove pruning wrappers to finalize model
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and hasattr(module, 'weight_orig'):
        prune.remove(module, "weight")

# # Load test data
with open("evaluation/test_set.json", "r") as f:
    test_data = json.load(f)
with open("evaluation/test_ground_truth.json", "r") as f:
    test_gt = json.load(f)

# Join QA with labels
# Clean and prepare questions and ground-truth labels
questions, y_true = [], []
for k in test_data:
    q = test_data[k]["QUESTION"]
    questions.append((q.strip() + ".") if not q.strip().endswith(".") else q.strip())
    y_true.append(test_gt[k])

latencies, gpu_util, mem_util, power, temp, y_pred = [], [], [], [], [], []

# Queries GPU statistics including utilization, memory, power, and temperature.
def get_gpu_info():
    res = subprocess.run([
        'nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,power.draw,temperature.gpu',
        '--format=csv,noheader,nounits'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    vals = res.stdout.decode('utf-8').strip().split(',')
    return {
        'gpu_utilization': float(vals[0]),
        'memory_utilization': float(vals[1]),
        'power_usage': float(vals[2]),
        'temperature': float(vals[3])
    }

# Inference on test set
model.eval()

for q in tqdm(questions):
    #format question
    q += " Answer in the following format in yes or no. the answer to the question given the context is"
    inputs = tokenizer(q, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    # Inference timing
    start = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=256, num_beams=1)
    torch.cuda.synchronize()
    latencies.append(time.time() - start)
    metrics = get_gpu_info()

    # Log GPU stats
    gpu_util.append(metrics['gpu_utilization'])
    mem_util.append(metrics['memory_utilization'])
    power.append(metrics['power_usage'])
    temp.append(metrics['temperature'])
    wandb.log({**metrics, "latency": latencies[-1]})

    # Decode answer
    # Parse model output to yes/no/maybe
    # Extract a clear yes/no/maybe answer from the decoded model output
    decoded = tokenizer.decode(out[0], skip_special_tokens=True).lower()
    if "yes" in decoded: y_pred.append("yes")
    elif "no" in decoded: y_pred.append("no")
    elif "maybe" in decoded: y_pred.append("maybe")
    else: y_pred.append("failed")

# Compute final metrics
accuracy = accuracy_score(y_true, y_pred)
thrpt = len(y_pred) / sum(latencies)

# Log summary metrics to wandb
wandb.log({
    "accuracy": accuracy,
    "throughput": thrpt,
    "avg_latency": sum(latencies)/len(latencies),
    "avg_gpu_utilization": sum(gpu_util)/len(gpu_util),
    "avg_memory_utilization": sum(mem_util)/len(mem_util),
    "avg_power_usage": sum(power)/len(power),
    "avg_temperature": sum(temp)/len(temp)
})
wandb.finish()
print(f"✅ Done! Accuracy: {accuracy:.4f}, Throughput: {thrpt:.2f} samples/sec")

