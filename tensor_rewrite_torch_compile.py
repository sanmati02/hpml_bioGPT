# bioGPT_compile_eval.py
import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import wandb
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import time

# ==== Argument Parser for Backend + Mode ==== #
parser = argparse.ArgumentParser()
parser.add_argument("--backend", type=str, default="inductor", help="torch.compile backend")
parser.add_argument("--mode", type=str, default="default", help="torch.compile mode")
parser.add_argument("--precision", type=str, default="float32", help="quantization")
args = parser.parse_args()

backend = args.backend
mode = args.mode
precision = args.precision

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

wandb.init(
    project="biogpt-pubmedqa",
    name=f"torch-compile-{backend}-{mode}-{precision}",
    config={"backend": backend, "mode": mode, "precision": precision, "task": "QA-PubMedQA"}
)

# ==== Load model and tokenizer ====
tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large-PubMEDQA")
model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large-PubMEDQA").to("cuda")
model.eval()

# ==== Apply torch.compile() with selected backend/mode ====
model = torch.compile(model, mode=mode, backend=backend)

# ==== Load test data ====
with open("test_set.json", "r") as f:
    test_data = json.load(f)
with open("test_ground_truth.json", "r") as f:
    test_gt = json.load(f)

data = []
for pmid, item in test_data.items():
    data.append({"question": item["QUESTION"], "label": test_gt[pmid]})
test_dataset = Dataset.from_list(data)

# ==== Preprocessing ====
def preprocess(example):
    prompt = example["question"].strip()
    if not prompt.endswith("."): prompt += "."
    full_prompt = prompt + " Answer in the following format in yes or no. the answer to the question given the context is"
    return tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)

def extract_answer(output):
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    if "yes" in decoded.lower(): return "yes"
    if "no" in decoded.lower(): return "no"
    if "maybe" in decoded.lower(): return "maybe"
    return "failed"

# ==== Inference Loop ====
inputs = [preprocess(ex) for ex in test_dataset]
y_true = [ex["label"] for ex in test_dataset]
y_pred, latencies = [], []

for input_ids in tqdm(inputs):
    input_ids = {k: v.to("cuda") for k, v in input_ids.items()}
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        output = model.generate(**input_ids, max_new_tokens=128, use_cache=False)
    torch.cuda.synchronize()
    latencies.append(time.time() - start)
    y_pred.append(extract_answer(output))

accuracy = accuracy_score(y_true, y_pred)
thrpt = len(y_pred) / sum(latencies)

wandb.log({
    "accuracy": accuracy,
    "throughput": thrpt,
    "avg_latency": sum(latencies)/len(latencies)
})

wandb.finish()
print(f"✅ Accuracy: {accuracy:.4f}, Throughput: {thrpt:.2f} samples/sec")
