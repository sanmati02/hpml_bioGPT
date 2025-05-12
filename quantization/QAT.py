"""
QAT_float16.py

Runs BioGPT-Large-PubMedQA for Quantization aware training
and logs inference metrics such as accuracy, latency, GPU stats using Weights & Biases.
"""

import torch
import torch.nn as nn
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
from sklearn.metrics import accuracy_score

def symmetric_quantize(tensor, num_bits=8):
    qmin = -2 ** (num_bits - 1)
    qmax = 2 ** (num_bits - 1) - 1
    max_val = tensor.abs().max()
    scale = max_val / qmax
    q_tensor = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    return q_tensor, scale

def asymmetric_quantize(tensor, num_bits=8):
    qmin = 0
    qmax = 2 ** num_bits - 1
    min_val = tensor.min()
    max_val = tensor.max()
    scale = (max_val - min_val) / (qmax - qmin + 1e-8)
    zero_point = torch.round(qmin - min_val / scale)
    q_tensor = torch.clamp(torch.round(tensor / scale + zero_point), qmin, qmax)
    return q_tensor, scale, zero_point

def dequantize(q_tensor, scale, zero_point=None):
    if zero_point is None:
        return q_tensor * scale  # symmetric
    else:
        return (q_tensor - zero_point) * scale  # asymmetric


# Load and preprocess PubMedQA training set
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

# Load model and optimizer 
config = AutoConfig.from_pretrained("microsoft/BioGPT-Large-PubMEDQA")
model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large-PubMEDQA", config=config)
model.gradient_checkpointing_enable()
model.to("cuda")
optimizer = AdamW(model.parameters(), lr=1e-5)

# AMP setup (Automatic Mixed Precision)
scaler = torch.amp.GradScaler()

# Training with float16 AMP
for epoch in range(10):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in loader:
        batch = {k: v.to("cuda") for k, v in batch.items()}
        optimizer.zero_grad()
        # Mixed precision context
        # with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        # Simulate symmetric quantization on weights
        for name, param in model.named_parameters():
            if "weight" in name and param.requires_grad:
                q_param, scale = symmetric_quantize(param.data)
                param.data = dequantize(q_param, scale)  # simulate dequantization after int8
        
        # Simulate asymmetric quantization on activations
        with torch.no_grad():
            for k in ["input_ids", "attention_mask"]:
                if k in batch:
                    q_act, scale, zp = asymmetric_quantize(batch[k])
                    batch[k] = dequantize(q_act, scale, zp).type_as(batch[k])
        
        out = model(**batch, use_cache=False)

        loss = out.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        print(loss.item())
        num_batches += 1
    avg_loss = total_loss / num_batches
    print(f"\U0001f4c9 Epoch {epoch+1} average loss: {avg_loss:.4f}")

# Inference on test set with Float16 
model.eval()

# Initialize WandB
wandb.init(
    project="biogpt-pubmedqa",
    name="qat-float16",
    config={"model": "BioGPT QAT AMP", "task": "QA-PubMedQA"}
)

# Load test data
with open("../evaluation/test_set.json", "r") as f:
    test_data = json.load(f)
with open("../evaluation/test_ground_truth.json", "r") as f:
    test_ground_truth = json.load(f)

# Join QA with labels
df = pd.DataFrame.from_dict(test_data, orient="index").reset_index().rename(columns={"index": "PMID"})
df_gt = pd.DataFrame.from_dict(test_ground_truth, orient="index").reset_index().rename(columns={"index": "PMID"})
df = pd.merge(df, df_gt, on="PMID")[["QUESTION", 0]].rename(columns={"QUESTION": "Question", 0: "Answer"})

# Clean and prepare questions and ground-truth labels
questions = [(q.strip() + "." if not q.strip().endswith(".") else q.strip()) for q in df["Question"].tolist()]
y_true = df["Answer"].tolist()

latencies, gpu_utilization_list, memory_utilization_list, power_usage_list, temperature_list = [], [], [], [], []

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

# Inference and logging
answers = []
for q in tqdm(questions):
    #format question
    q += " Answer in the following format in yes or no. the answer to the question given the context is"
    inputs = tokenizer(q, return_tensors="pt").to('cuda')
    # Inference timing
    start_time = time.time()
    with torch.inference_mode():
         # Float16 inference with no_grad context
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            output = model.generate(**inputs, max_new_tokens=256, num_beams = 1, early_stopping = False, do_sample = False)
    latency = time.time() - start_time
    
    # Decode answer
    answers.append(tokenizer.decode(output[0], skip_special_tokens=True))
    latencies.append(latency)

    # Log GPU stats
    metrics = get_gpu_info()
    wandb.log({"latency": latency, **metrics})
    gpu_utilization_list.append(metrics['gpu_utilization'])
    memory_utilization_list.append(metrics['memory_utilization'])
    power_usage_list.append(metrics['power_usage'])
    temperature_list.append(metrics['temperature'])


# Remove leading phrases or noise that often precede the actual answer
prefixes = ['(learned[0-9]+ )+', 'we can conclude that', 'we have that', 'in conclusion,']
def strip_prefix(line):
    for p in prefixes:
        if re.search(p, line): line = re.split(p, line)[-1].strip(); break
    return line

# Parse model output to yes/no/maybe
# Extract a clear yes/no/maybe answer from the decoded model output
def convert_ans(s): return re.search(r"the answer to the question given the context is (yes|no|maybe)\b", s, re.I)
hypothesis = []
for line in answers:
    if line.endswith("."): line = line[:-1]
    ans_match = convert_ans(strip_prefix(line))
    hypothesis.append(ans_match.groups()[0].strip() if ans_match else "failed")

# Compute final metrics
accuracy = accuracy_score(y_true[:len(hypothesis)], hypothesis)
throughput = len(questions) / sum(latencies)

# Log summary metrics to wandb
wandb.log({
    "throughput": throughput,
    "avg_latency": sum(latencies)/len(latencies),
    "accuracy": accuracy,
    "avg_gpu_utilization": sum(gpu_utilization_list)/len(gpu_utilization_list),
    "avg_memory_utilization": sum(memory_utilization_list)/len(memory_utilization_list),
    "avg_power_usage": sum(power_usage_list)/len(power_usage_list),
    "avg_temperature": sum(temperature_list)/len(temperature_list)
})
wandb.finish()

