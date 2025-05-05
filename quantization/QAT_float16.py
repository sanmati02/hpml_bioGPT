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

# ==== 1. Load and preprocess PubMedQA ====
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

train_data = Dataset.from_list(examples)
tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large-PubMEDQA")

def preprocess(example):
    prompt = example["question"] + " Context: " + example["context"] + ". Answer in yes or no:"
    full_text = prompt + " " + example["label"]
    enc = tokenizer(full_text, padding="max_length", truncation=True, max_length=256)
    prompt_ids = tokenizer(prompt, padding="max_length", truncation=True, max_length=256)["input_ids"]
    enc["labels"] = [-100 if i < sum(p != tokenizer.pad_token_id for p in prompt_ids) else t for i, t in enumerate(enc["input_ids"])]
    return enc

train_data = train_data.map(preprocess)

def collate_fn(batch):
    tensor_keys = ["input_ids", "attention_mask", "labels"]
    return {k: torch.tensor([f[k] for f in batch]) for k in tensor_keys if k in batch[0]}

loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn)

# ==== 2. Load model and optimizer ====
config = AutoConfig.from_pretrained("microsoft/BioGPT-Large-PubMEDQA")
#config.num_hidden_layers = 10
model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large-PubMEDQA", config=config)
model.gradient_checkpointing_enable()
model.to("cuda")
optimizer = AdamW(model.parameters(), lr=1e-5)

# ==== 3. AMP setup ====
scaler = torch.amp.GradScaler()

# ==== 4. Training with float16 AMP ====
for epoch in range(10):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in loader:
        batch = {k: v.to("cuda") for k, v in batch.items()}
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
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

# ==== 5. Inference on test set ====
model.eval()

wandb.init(
    project="biogpt-pubmedqa",
    name="qat-float16",
    config={"model": "BioGPT QAT AMP", "task": "QA-PubMedQA"}
)

with open("../evaluation/test_set.json", "r") as f:
    test_data = json.load(f)
with open("../evaluation/test_ground_truth.json", "r") as f:
    test_ground_truth = json.load(f)

df = pd.DataFrame.from_dict(test_data, orient="index").reset_index().rename(columns={"index": "PMID"})
df_gt = pd.DataFrame.from_dict(test_ground_truth, orient="index").reset_index().rename(columns={"index": "PMID"})
df = pd.merge(df, df_gt, on="PMID")[["QUESTION", 0]].rename(columns={"QUESTION": "Question", 0: "Answer"})

questions = [(q.strip() + "." if not q.strip().endswith(".") else q.strip()) for q in df["Question"].tolist()]
y_true = df["Answer"].tolist()

latencies, gpu_utilization_list, memory_utilization_list, power_usage_list, temperature_list = [], [], [], [], []

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

answers = []
questions = questions # demo only
for q in tqdm(questions):
    q += " Answer in the following format in yes or no. the answer to the question given the context is"
    inputs = tokenizer(q, return_tensors="pt").to('cuda')
    start_time = time.time()
    with torch.inference_mode():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            output = model.generate(**inputs, max_new_tokens=256, num_beams = 1, early_stopping = False, do_sample = False)
    latency = time.time() - start_time
    metrics = get_gpu_info()
    wandb.log({"latency": latency, **metrics})
    answers.append(tokenizer.decode(output[0], skip_special_tokens=True))
    latencies.append(latency)
    gpu_utilization_list.append(metrics['gpu_utilization'])
    memory_utilization_list.append(metrics['memory_utilization'])
    power_usage_list.append(metrics['power_usage'])

    temperature_list.append(metrics['temperature'])



prefixes = ['(learned[0-9]+ )+', 'we can conclude that', 'we have that', 'in conclusion,']
def strip_prefix(line):
    for p in prefixes:
        if re.search(p, line): line = re.split(p, line)[-1].strip(); break
    return line

def convert_ans(s): return re.search(r"the answer to the question given the context is (yes|no|maybe)\b", s, re.I)
hypothesis = []
for line in answers:
    if line.endswith("."): line = line[:-1]
    ans_match = convert_ans(strip_prefix(line))
    hypothesis.append(ans_match.groups()[0].strip() if ans_match else "failed")

accuracy = accuracy_score(y_true[:len(hypothesis)], hypothesis)
#print(accuracy)
throughput = len(questions) / sum(latencies)
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

