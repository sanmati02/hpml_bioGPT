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

# 🟡 Initialize Weights & Biases
wandb.init(
    project="biogpt-pubmedqa",
    name="baseline-inference",
    config={
        "model": "BioGPT-Large-PubMedQA",
        "num_beams": 1,
        "max_new_tokens": 256,
        "task": "QA-PubMedQA",
    }
)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large-PubMedQA")
model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large-PubMedQA").cuda()

# Load test data
with open("../evaluation/test_set.json", "r") as f:
    test_data = json.load(f)

df = pd.DataFrame.from_dict(test_data, orient="index")
df.reset_index(inplace=True)
df.rename(columns={"index": "PMID"}, inplace=True)

with open("../evaluation/test_ground_truth.json", "r") as f:
    test_ground_truth = json.load(f)

df_test_ground_truth = pd.DataFrame.from_dict(test_ground_truth, orient="index")
df_test_ground_truth.reset_index(inplace=True)
df_test_ground_truth.rename(columns={"index": "PMID"}, inplace=True)

test_data = pd.merge(df_test_ground_truth, df, on="PMID", how="inner")
test_data = test_data[['QUESTION', 0]]
test_data.columns = ['Question', 'Answer']

# Preprocess questions
questions = []
for sentence in test_data["Question"]:
    sentence = sentence.replace('\n', '').strip()
    if not sentence.endswith("."):
        sentence = sentence + "."
    questions.append(sentence)

y_true = test_data["Answer"].tolist()

# Containers for metrics
latencies = []
gpu_utilization_list = []
memory_utilization_list = []
power_usage_list = []
temperature_list = []

# Function to get GPU metrics
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

# Inference loop
answers = []
for question in tqdm(questions):
    question = question + ". Answer in the following format in yes or no. the answer to the question given the context is"
    start_time = time.time()

    inputs = tokenizer(question, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        beam_output = model.generate(**inputs,
                                     max_new_tokens=256,
                                     num_beams=1,
                                     early_stopping=False,
                                     do_sample=False)
        latency = time.time() - start_time
        latencies.append(latency)

        gpu_metrics = get_gpu_info()
        gpu_utilization_list.append(gpu_metrics['gpu_utilization'])
        memory_utilization_list.append(gpu_metrics['memory_utilization'])
        power_usage_list.append(gpu_metrics['power_usage'])
        temperature_list.append(gpu_metrics['temperature'])

        # 🟢 Log to wandb per-inference
        wandb.log({
            "latency": latency,
            "gpu_utilization": gpu_metrics['gpu_utilization'],
            "memory_utilization": gpu_metrics['memory_utilization'],
            "power_usage": gpu_metrics['power_usage'],
            "temperature": gpu_metrics['temperature']
        })

        answers.append(tokenizer.decode(beam_output[0], skip_special_tokens=True))

# Postprocess output
prefix = [
    '(learned[0-9]+ )+',
    'we can conclude that',
    'we have that',
    'in conclusion,',
]

def strip_prefix(line):
    for p in prefix:
        res = re.search(p, line)
        if res is not None:
            line = re.split(p, line)[-1].strip()
            break
    return line

def convert_relis_sentence(sentence):
    ans = None
    segs = re.search(r"the answer to the question given the context is (yes|no|maybe)\b", sentence, re.IGNORECASE)
    if segs is not None:
        segs = segs.groups()
        ans = segs[0].strip()
    return ans

hypothesis = []
fail_cnt = 0

for i, line in enumerate(answers):
    if line[-1] == ".":
        line = line[:-1]
    strip_line = strip_prefix(line)
    ans = convert_relis_sentence(strip_line)
    if ans is not None:
        hypothesis.append(ans)
    else:
        hypothesis.append("failed")
        fail_cnt += 1
        print("Failed:id:{}, line:{}".format(i+1, line))

# Final evaluation
total_time = sum(latencies)
throughput = len(questions) / total_time
avg_latency = sum(latencies) / len(latencies)
accuracy = accuracy_score(y_true, hypothesis)
avg_gpu_utilization = sum(gpu_utilization_list) / len(gpu_utilization_list)
avg_memory_utilization = sum(memory_utilization_list) / len(memory_utilization_list)
avg_power_usage = sum(power_usage_list) / len(power_usage_list)
avg_temperature = sum(temperature_list) / len(temperature_list)

# Save output
with open('my_list.json', 'w') as file:
    json.dump(hypothesis, file)

# Print metrics
print(f"Throughput: {throughput:.2f} questions per second")
print(f"Average Latency per inference: {avg_latency:.4f} seconds")
print(f"Accuracy: {accuracy}")
print(f"Average GPU Utilization: {avg_gpu_utilization:.2f}%")
print(f"Average Memory Utilization: {avg_memory_utilization:.2f}%")
print(f"Average Power Usage: {avg_power_usage:.2f} W")
print(f"Average Temperature: {avg_temperature:.2f} °C")

# 🟢 Final wandb log
wandb.log({
    "throughput": throughput,
    "avg_latency": avg_latency,
    "accuracy": accuracy,
    "avg_gpu_utilization": avg_gpu_utilization,
    "avg_memory_utilization": avg_memory_utilization,
    "avg_power_usage": avg_power_usage,
    "avg_temperature": avg_temperature
})

wandb.finish()

