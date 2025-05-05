import subprocess
import time
import torch
import numpy as np
import json
import pandas as pd
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import wandb
import sys
import onnx
from transformers import AutoTokenizer
from pyinfinitensor.onnx import OnnxStub, backend

# Initialize wandb
wandb.init(
    project="biogpt-pubmedqa",
    name="int8-einnet-inference",
    config={"model": "BioGPT-Large-PubMedQA", "precision": "int8", "task": "QA-PubMedQA"}
)

# Model Path
onnx_model_path = "biogpt_large_pubmedqa.onnx"  # Update with your actual ONNX model path

# Load ONNX model
onnx_model = onnx.load(onnx_model_path)
onnx_input = onnx_model.graph.input[0]
input_shape = [[d.dim_value for d in _input.type.tensor_type.shape.dim] for _input in onnx_model.graph.input]
input_shape = input_shape[0]

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large-PubMedQA")

# Load test data
with open("test_set.json", "r") as f:
    test_data = json.load(f)
df = pd.DataFrame.from_dict(test_data, orient="index").reset_index().rename(columns={"index": "PMID"})

with open("test_ground_truth.json", "r") as f:
    test_gt = json.load(f)
gt_df = pd.DataFrame.from_dict(test_gt, orient="index").reset_index().rename(columns={"index": "PMID"})
test_data = pd.merge(gt_df, df, on="PMID", how="inner")[["QUESTION", 0]]
test_data.columns = ["Question", "Answer"]

# Preprocess questions
questions = [(q.strip() + ".") if not q.strip().endswith(".") else q.strip() for q in test_data["Question"]]
y_true = test_data["Answer"].tolist()

# Setup metrics
latencies, answers = [], []
gpu_utilization_list, memory_utilization_list, power_usage_list, temperature_list = [], [], [], []

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

# Run inference on each question
for question in tqdm(questions):
    question += " Answer in the following format in yes or no. the answer to the question given the context is"
    start_time = time.time()
    
    # Tokenize input
    inputs = tokenizer(question, return_tensors="pt").to("cuda")
    
    # Prepare input for ONNX model
    input_data = np.random.random(input_shape).astype(np.float32)  # This should be replaced with actual tokenized data
    
    # Initialize the ONNX model for inference
    model = OnnxStub(onnx_model, backend.cuda_runtime())
    next(iter(model.inputs.values())).copyin_numpy(input_data)
    
    # Run the model
    model.run()
    outputs = next(iter(model.outputs.values())).copyout_numpy()
    outputs = torch.tensor(outputs)

    latency = time.time() - start_time
    latencies.append(latency)

    # Collect GPU stats
    gpu_metrics = get_gpu_info()
    gpu_utilization_list.append(gpu_metrics['gpu_utilization'])
    memory_utilization_list.append(gpu_metrics['memory_utilization'])
    power_usage_list.append(gpu_metrics['power_usage'])
    temperature_list.append(gpu_metrics['temperature'])

    wandb.log({**gpu_metrics, "latency": latency})
    answers.append(outputs)  # Modify if needed to get actual answer from output

# Postprocessing: handle prefixes, answer extraction, and accuracy calculation
prefix = ['(learned[0-9]+ )+', 'we can conclude that', 'we have that', 'in conclusion,']

def strip_prefix(line):
    for p in prefix:
        if re.search(p, line):
            return re.split(p, line)[-1].strip()
    return line

def convert_relis_sentence(sentence):
    match = re.search(r"the answer to the question given the context is (yes|no|maybe)", sentence, re.IGNORECASE)
    return match.group(1).strip() if match else None

hypothesis, fail_cnt = [], 0
for i, line in enumerate(answers):
    line = line[:-1] if line.endswith(".") else line
    ans = convert_relis_sentence(strip_prefix(line))
    hypothesis.append(ans if ans else "failed")
    if not ans:
        fail_cnt += 1
        print(f"Failed:id:{i+1}, line:{line}")

accuracy = accuracy_score(y_true, hypothesis)
total_time = sum(latencies)
avg_latency = total_time / len(latencies)
throughput = len(questions) / total_time

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

# Save output
with open('int8_output.json', 'w') as f:
    json.dump(hypothesis, f)

print(f"Accuracy: {accuracy:.4f}, Avg Latency: {avg_latency:.4f}, Throughput: {throughput:.2f}")
