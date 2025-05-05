"""
ptq_dynamic_int8.py

Runs BioGPT-Large-PubMedQA for Post training quantization (int8)
and logs inference metrics such as accuracy, latency, GPU stats using Weights & Biases.
"""
import subprocess
import time
from tqdm import tqdm
import torch
from torch.quantization import quantize_dynamic
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pandas as pd
import re
from sklearn.metrics import accuracy_score
import wandb

# Initialize WandB
wandb.init(
    project="biogpt-pubmedqa",
    name="int8-dynamic-inference",
    config={"model": "BioGPT-Large-PubMedQA", "precision": "int8-dynamic", "task": "QA-PubMedQA"}
)

# Load pretrained model and apply dynamic quantization to linear layers
model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large-PubMedQA")
model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8).eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large-PubMedQA")

# Load test data
with open("../evaluation/test_set.json", "r") as f:
    test_data = json.load(f)
df = pd.DataFrame.from_dict(test_data, orient="index").reset_index().rename(columns={"index": "PMID"})
with open("../evaluation/test_ground_truth.json", "r") as f:
    test_gt = json.load(f)

# Join QA with labels
gt_df = pd.DataFrame.from_dict(test_gt, orient="index").reset_index().rename(columns={"index": "PMID"})
test_data = pd.merge(gt_df, df, on="PMID", how="inner")[["QUESTION", 0]]
test_data.columns = ["Question", "Answer"]

# Clean and prepare questions and ground-truth labels
questions = [(q.strip() + ".") if not q.strip().endswith(".") else q.strip() for q in test_data["Question"]]
y_true = test_data["Answer"].tolist()

latencies, answers = [], []
gpu_utilization_list, memory_utilization_list, power_usage_list, temperature_list = [], [], [], []

# Queries GPU statistics including utilization, memory, power, and temperature.
def get_gpu_info():
    result = subprocess.run([
        'nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,power.draw,temperature.gpu',
        '--format=csv,noheader,nounits'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    gpu_info = result.stdout.decode('utf-8').strip().split(',')
    return {
        'gpu_utilization': float(gpu_info[0]),
        'memory_utilization': float(gpu_info[1]),
        'power_usage': float(gpu_info[2]),
        'temperature': float(gpu_info[3])
    }

# Inference and logging
for question in tqdm(questions):
    # Format input
    question += " Answer in the following format in yes or no. the answer to the question given the context is"
    
    # Inference timing
    start_time = time.time()
    inputs = tokenizer(question, return_tensors="pt")
    with torch.inference_mode():
        # Run model inference with no gradient tracking (saves memory and improves speed)
        output = model.generate(**inputs, max_new_tokens=256, num_beams=1)
    latency = time.time() - start_time
    latencies.append(latency)

    # Log GPU stats
    gpu_metrics = get_gpu_info()
    gpu_utilization_list.append(gpu_metrics['gpu_utilization'])
    memory_utilization_list.append(gpu_metrics['memory_utilization'])
    power_usage_list.append(gpu_metrics['power_usage'])
    temperature_list.append(gpu_metrics['temperature'])

    # Decode answer
    wandb.log({**gpu_metrics, "latency": latency})
    answers.append(tokenizer.decode(output[0], skip_special_tokens=True))

prefix = ['(learned[0-9]+ )+', 'we can conclude that', 'we have that', 'in conclusion,']

# Remove leading phrases or noise that often precede the actual answer
def strip_prefix(line):
    for p in prefix:
        if re.search(p, line):
            return re.split(p, line)[-1].strip()
    return line

# Extract a clear yes/no/maybe answer from the decoded model output
def convert_relis_sentence(sentence):
    match = re.search(r"the answer to the question given the context is (yes|no|maybe)", sentence, re.IGNORECASE)
    return match.group(1).strip() if match else None

# Parse model output to yes/no/maybe
hypothesis, fail_cnt = [], 0
for i, line in enumerate(answers):
    line = line[:-1] if line.endswith(".") else line
    ans = convert_relis_sentence(strip_prefix(line))
    hypothesis.append(ans if ans else "failed")
    if not ans:
        fail_cnt += 1
        print(f"Failed:id:{i+1}, line:{line}")

# Compute final metrics
accuracy = accuracy_score(y_true, hypothesis)
total_time = sum(latencies)
avg_latency = total_time / len(latencies)
throughput = len(questions) / total_time

# Log summary metrics to wandb
wandb.log({
    "accuracy": accuracy,
    "throughput": throughput,
    "avg_latency": avg_latency,
    "avg_gpu_utilization": sum(gpu_utilization_list) / len(gpu_utilization_list),
    "avg_memory_utilization": sum(memory_utilization_list) / len(memory_utilization_list),
    "avg_power_usage": sum(power_usage_list) / len(power_usage_list),
    "avg_temperature": sum(temperature_list) / len(temperature_list)
})

wandb.finish()

with open('int8_dynamic_output.json', 'w') as f:
    json.dump(hypothesis, f)

print(f"Accuracy: {accuracy:.4f}, Avg Latency: {avg_latency:.4f}, Throughput: {throughput:.2f}")

