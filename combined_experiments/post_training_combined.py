"""
float16_compile_experiment.py

Runs BioGPT-Large-PubMedQA using float16 precision
+ either AOT Eager or InductorMax Autotune backends.
Logs accuracy, latency, and GPU metrics via Weights & Biases.
"""

import torch
import argparse
import subprocess
import wandb
import time, json
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Argparse for backend ===
parser = argparse.ArgumentParser()
parser.add_argument("--backend", type=str, choices=["aot_eager", "inductor"], required=True, help="torch.compile backend")
args = parser.parse_args()
backend = args.backend

# === Set compile mode for InductorMax Autotune ===
compile_args = {"backend": backend}
if backend == "inductor":
    compile_args["mode"] = "max-autotune"

# === Initialize wandb ===
wandb.init(
    project="biogpt-pubmedqa",
    name=f"float16-{backend}",
    config={"precision": "float16", "backend": backend}
)

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large-PubMEDQA")
model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large-PubMEDQA").half().cuda().eval()
model = torch.compile(model, **compile_args)

# === Load test data ===
with open("../evaluation/test_set.json", "r") as f:
    test_data = json.load(f)
with open("../evaluation/test_ground_truth.json", "r") as f:
    test_gt = json.load(f)

questions = [(v["QUESTION"].strip() + "." if not v["QUESTION"].strip().endswith(".") else v["QUESTION"].strip()) for v in test_data.values()]
y_true = list(test_gt.values())

# === GPU stat collector ===
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

# === Inference ===
latencies, answers = [], []
gpu_utilization_list, memory_utilization_list, power_usage_list, temperature_list = [], [], [], []

for q in tqdm(questions):
    q += " Answer in the following format in yes or no. the answer to the question given the context is"
    inputs = tokenizer(q, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=256)
    torch.cuda.synchronize()
    latencies.append(time.time() - start_time)

    metrics = get_gpu_info()
    gpu_utilization_list.append(metrics["gpu_utilization"])
    memory_utilization_list.append(metrics["memory_utilization"])
    power_usage_list.append(metrics["power_usage"])
    temperature_list.append(metrics["temperature"])
    wandb.log({**metrics, "latency": latencies[-1]})

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    if "yes" in decoded.lower(): answers.append("yes")
    elif "no" in decoded.lower(): answers.append("no")
    elif "maybe" in decoded.lower(): answers.append("maybe")
    else: answers.append("failed")

# === Metrics ===
accuracy = accuracy_score(y_true[:len(answers)], answers)
throughput = len(answers) / sum(latencies)

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
print(f"✅ Done. Accuracy: {accuracy:.4f}, Throughput: {throughput:.2f}/s")

