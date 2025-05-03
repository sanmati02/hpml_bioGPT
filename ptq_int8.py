# biogpt_trt_int8_eval.py
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
import torch_tensorrt

# ==== 1. Load and preprocess PubMedQA dev set for calibration ====
with open("../pubmedqa/data/pqal_fold0/dev_set.json") as f:
    raw_data = json.load(f)
examples = []
for pmid, item in raw_data.items():
    examples.append({
        "PMID": pmid,
        "question": item["QUESTION"],
        "context": " ".join(item["CONTEXTS"]),
        "label": item["final_decision"]
    })

tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large-PubMEDQA")

def preprocess(example):
    prompt = example["question"] + " Context: " + example["context"] + ". Answer in yes or no:"
    return tokenizer(prompt, padding="max_length", truncation=True, max_length=256)

dev_data = Dataset.from_list(examples).map(preprocess)

def calibrate_model_inputs():
    for item in dev_data:
        input_tensor = torch.tensor(item["input_ids"]).unsqueeze(0).to(torch.int32).cuda()
        yield (input_tensor,)

# ==== 2. Load and convert model to TorchScript ====
model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large-PubMEDQA").cuda().eval()
scripted_model = torch.jit.script(model)

# ==== 3. Compile with TensorRT INT8 ====
trt_model = torch_tensorrt.compile(
    scripted_model,
    inputs=[torch_tensorrt.Input((1, 256), dtype=torch.int32)],
    enabled_precisions={torch.int8},
    calibrator=calibrate_model_inputs(),
    truncate_long_and_double=True
)

# ==== 4. Load test data ====
with open("test_set.json", "r") as f:
    test_data = json.load(f)
df = pd.DataFrame.from_dict(test_data, orient="index").reset_index().rename(columns={"index": "PMID"})
with open("test_ground_truth.json", "r") as f:
    test_gt = json.load(f)
gt_df = pd.DataFrame.from_dict(test_gt, orient="index").reset_index().rename(columns={"index": "PMID"})
test_data = pd.merge(gt_df, df, on="PMID", how="inner")[["QUESTION", 0]]
test_data.columns = ["Question", "Answer"]

questions = [(q.strip() + ".") if not q.strip().endswith(".") else q.strip() for q in test_data["Question"]]
y_true = test_data["Answer"].tolist()

# ==== 5. Inference Loop + Metrics ====
latencies, answers = [], []
gpu_utilization_list, memory_utilization_list, power_usage_list, temperature_list = [], [], [], []

wandb.init(
    project="biogpt-pubmedqa",
    name="tensorrt-int8-inference",
    config={"model": "BioGPT-Large-PubMedQA", "precision": "int8-tensorrt", "task": "QA-PubMedQA"}
)

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

def convert_relis_sentence(sentence):
    match = re.search(r"the answer to the question given the context is (yes|no|maybe)", sentence, re.IGNORECASE)
    return match.group(1).strip() if match else None

for question in tqdm(questions):
    prompt = question + " Answer in the following format in yes or no. the answer to the question given the context is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    start = time.time()
    with torch.no_grad():
        output = trt_model.generate(input_ids=input_ids, max_new_tokens=256, num_beams=1)
    latency = time.time() - start
    latencies.append(latency)

    gpu_metrics = get_gpu_info()
    gpu_utilization_list.append(gpu_metrics['gpu_utilization'])
    memory_utilization_list.append(gpu_metrics['memory_utilization'])
    power_usage_list.append(gpu_metrics['power_usage'])
    temperature_list.append(gpu_metrics['temperature'])

    wandb.log({**gpu_metrics, "latency": latency})
    answers.append(tokenizer.decode(output[0], skip_special_tokens=True))

# ==== 6. Postprocessing and Logging ====
prefix = ['(learned[0-9]+ )+', 'we can conclude that', 'we have that', 'in conclusion,']

def strip_prefix(line):
    for p in prefix:
        if re.search(p, line):
            return re.split(p, line)[-1].strip()
    return line

hypothesis = []
for line in answers:
    line = line[:-1] if line.endswith(".") else line
    ans = convert_relis_sentence(strip_prefix(line))
    hypothesis.append(ans if ans else "failed")

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

with open('tensorrt_int8_output.json', 'w') as f:
    json.dump(hypothesis, f)

print(f"Accuracy: {accuracy:.4f}, Avg Latency: {avg_latency:.4f}, Throughput: {throughput:.2f}")

