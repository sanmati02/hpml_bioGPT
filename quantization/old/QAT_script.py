# qat_biogpt_pubmedqa.py
import torch
import torch.nn as nn
import torch.nn.qat as nnqat
import torch.quantization as tq
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from datasets import Dataset
import json
import wandb

# ==== 1. Load and preprocess PubMedQA ====
with open("../../pubmedqa/data/pqal_fold0/dev_set.json") as f:
    raw_data = json.load(f)
examples = []
for pmid, item in raw_data.items():
    examples.append({
        "PMID": pmid,
        "question": item["QUESTION"],
        "context": " ".join(item["CONTEXTS"]),
        "label": item["final_decision"]
    })
train_data = Dataset.from_list(examples[:10])  # small subset for dev

tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large-PubMEDQA")

# ==== 2. Preprocess function ====
def preprocess(example):
    prompt = example["question"] + " Context: " + example["context"] + ". Answer in yes or no:"
    full_text = prompt + " " + example["label"]
    enc = tokenizer(full_text, padding="max_length", truncation=True, max_length=96)
    prompt_ids = tokenizer(prompt, padding="max_length", truncation=True, max_length=96)["input_ids"]
    enc["labels"] = [-100 if i < sum(p != tokenizer.pad_token_id for p in prompt_ids) else t for i, t in enumerate(enc["input_ids"])]
    return enc

train_data = train_data.map(preprocess)

def collate_fn(batch):
    tensor_keys = ["input_ids", "attention_mask", "labels"]
    return {k: torch.tensor([f[k] for f in batch]) for k in tensor_keys if k in batch[0]}

loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collate_fn)

# ==== 3. Load model and patch for QAT ====
def replace_linear_with_qat(module, qconfig):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            qat_linear = nnqat.Linear(child.in_features, child.out_features, bias=child.bias is not None, qconfig=qconfig)
            qat_linear.weight.data = child.weight.data.clone()
            if child.bias is not None:
                qat_linear.bias.data = child.bias.data.clone()
            setattr(module, name, qat_linear)
        elif isinstance(child, nn.GELU):
            setattr(module, name, nn.ReLU())  # more QAT-friendly
        else:
            replace_linear_with_qat(child, qconfig)

model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large-PubMEDQA")
model.qconfig = tq.get_default_qat_qconfig("fbgemm")

model.train()
replace_linear_with_qat(model, model.qconfig)
model = tq.prepare_qat(model, inplace=True).to("cuda")

# ==== 4. Training ====
wandb.init(project="biogpt-pubmedqa", name="qat-gpu-train")
model.train()
optimizer = AdamW(model.parameters(), lr=3e-5)

for epoch in range(2):
    for batch in loader:
        batch = {k: v.to("cuda") for k, v in batch.items()}
        out = model(**batch, use_cache=False)
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        wandb.log({"loss": loss.item()})

# ==== 5. Save checkpoint ====
torch.save(model.state_dict(), "biogpt_qat_pubmedqa.pt")
print("✅ Saved QAT-trained BioGPT model.")

