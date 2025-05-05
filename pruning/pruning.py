"""
pruning.py

Runs BioGPT-Large-PubMedQA with unstructred pruning. 
Logs inference metrics such as accuracy, latency, GPU stats using Weights & Biases.
"""

import os
import time
import json
import torch
import argparse
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.nn.utils.prune as prune
import wandb

# 🟡 Initialize wandb
wandb.init(
    project="biogpt-pubmedqa",
    name="pruning_weights_eval",
    config={"model": "BioGPT-Large-PubMedQA", "precision": "float16", "task": "QA-PubMedQA", "pruning": "L1 unstructured"}
)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large-PubMedQA")
model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large-PubMedQA").half().cuda().eval()

# Apply Unstructured L1-Norm Pruning to All Linear Layers 
def apply_pruning(model, amount=0.2):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)

# Apply pruning at the defined sparsity level
apply_pruning(model)

# Load test data
with open("../evaluation/test_set.json", "r") as f:
    test_data = json.load(f)
with open("../evaluation/test_ground_truth.json", "r") as f:
    test_gt = json.load(f)

# Join QA with labels
data = [{"question": item["QUESTION"], "label": test_gt[pmid]} for pmid, item in test_data.items()]
test_dataset = Dataset.from_list(data)

# Tokenizes the input question and context into a prompt format suitable for generation.
def preprocess(example):
    prompt = example["question"].strip()
    if not prompt.endswith("."): prompt += "."
    full_prompt = prompt + " Answer in the following format in yes or no. the answer to the question given the context is"
    return tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)

#  Decodes generated tokens and extracts the predicted answer from the output string.
def extract_answer(output):
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    decoded = decoded.lower()
    if "yes" in decoded: return "yes"
    if "no" in decoded: return "no"
    if "maybe" in decoded: return "maybe"
    return "failed"

# Queries GPU statistics including utilization, memory, power, and temperature.
def get_gpu_info():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,power.draw,temperature.gpu', '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    gpu_info = result.stdout.decode('utf-8').strip().split(',')
    return {
        'gpu_utilization': float(gpu_info[0]),
        'memory_utilization': float(gpu_info[1]),
        'power_usage': float(gpu_info[2]),
        'temperature': float(gpu_info[3])
    }

# Preprocess inputs and prepare label list
inputs = [preprocess(ex) for ex in test_dataset]
y_true = [ex["label"] for ex in test_dataset]
y_pred = []
latencies = []
gpu_utilization_list = []
memory_utilization_list = []
power_usage_list = []
temperature_list = []

# Loop through each preprocessed input and perform inference
for input_ids in tqdm(inputs):
    # Move inputs to GPU
    input_ids = {k: v.to("cuda") for k, v in input_ids.items()}
    
    # Sync GPU before and after inference for accurate latency measurement
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        output = model.generate(**input_ids, max_new_tokens=256)
    torch.cuda.synchronize()
    latency = time.time() - start
    latencies.append(latency)

   # Decode model output and append to predictions
    answer = extract_answer(output)
    y_pred.append(answer)

    # Collect and store GPU stats for each inference step
    gpu_metrics = get_gpu_info()
    gpu_utilization_list.append(gpu_metrics['gpu_utilization'])
    memory_utilization_list.append(gpu_metrics['memory_utilization'])
    power_usage_list.append(gpu_metrics['power_usage'])
    temperature_list.append(gpu_metrics['temperature'])

    # Log GPU stats
    wandb.log({
        "latency": latency,
        "gpu_utilization": gpu_metrics['gpu_utilization'],
        "memory_utilization": gpu_metrics['memory_utilization'],
        "power_usage": gpu_metrics['power_usage'],
        "temperature": gpu_metrics['temperature']
    })

# Compute final metrics
accuracy = accuracy_score(y_true, y_pred)
throughput = len(y_pred) / sum(latencies)
avg_latency = sum(latencies) / len(latencies)
avg_gpu_utilization = sum(gpu_utilization_list) / len(gpu_utilization_list)
avg_memory_utilization = sum(memory_utilization_list) / len(memory_utilization_list)
avg_power_usage = sum(power_usage_list) / len(power_usage_list)
avg_temperature = sum(temperature_list) / len(temperature_list)

# Log summary metrics to wandb
wandb.log({
    "accuracy": accuracy,
    "throughput": throughput,
    "avg_latency": avg_latency,
    "avg_gpu_utilization": avg_gpu_utilization,
    "avg_memory_utilization": avg_memory_utilization,
    "avg_power_usage": avg_power_usage,
    "avg_temperature": avg_temperature
})
wandb.finish()

print(f"✅ Accuracy: {accuracy:.4f}")
print(f"Throughput: {throughput:.2f} samples/sec")
print(f"Average Latency: {avg_latency:.4f} sec")
print(f"Avg GPU Utilization: {avg_gpu_utilization:.2f}%")
print(f"Avg Memory Utilization: {avg_memory_utilization:.2f}%")
print(f"Avg Power Usage: {avg_power_usage:.2f} W")
print(f"Avg Temperature: {avg_temperature:.2f} °C")


