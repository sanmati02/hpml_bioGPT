# quant_fp16_baseline_eval.py — Float16 Evaluation Script
import subprocess
import time
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pandas as pd
import re
from sklearn.metrics import accuracy_score
import wandb

wandb.init(
    project="biogpt-pubmedqa",
    name="fp16-inference",
    config={"model": "BioGPT-Large-PubMedQA", "precision": "float16", "task": "QA-PubMedQA"}
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large-PubMedQA")
model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large-PubMedQA").half().cuda().eval()

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

for question in tqdm(questions):
    question += " Answer in the following format in yes or no. the answer to the question given the context is"
    start_time = time.time()
    inputs = tokenizer(question, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        inputs["input_ids"] = inputs["input_ids"].to("cuda")  # leave as int64
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].half().to("cuda")

        output = model.generate(**inputs, max_new_tokens=256, num_beams=1)
    latency = time.time() - start_time
    latencies.append(latency)

    gpu_metrics = get_gpu_info()
    gpu_utilization_list.append(gpu_metrics['gpu_utilization'])
    memory_utilization_list.append(gpu_metrics['memory_utilization'])
    power_usage_list.append(gpu_metrics['power_usage'])
    temperature_list.append(gpu_metrics['temperature'])

    wandb.log({**gpu_metrics, "latency": latency})
    answers.append(tokenizer.decode(output[0], skip_special_tokens=True))

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

with open('fp16_output.json', 'w') as f:
    json.dump(hypothesis, f)

print(f"Accuracy: {accuracy:.4f}, Avg Latency: {avg_latency:.4f}, Throughput: {throughput:.2f}")
