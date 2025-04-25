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
from tvm import auto_scheduler
import tvm

# ==== Argument Parser for Backend + Mode ==== #
parser = argparse.ArgumentParser()
parser.add_argument("--backend", type=str, default="inductor", help="torch.compile backend")
parser.add_argument("--mode", type=str, default="default", help="torch.compile mode")
parser.add_argument("--precision", type=str, default="float32", help="quantization")
args = parser.parse_args()

backend = "tvm"
mode = "ansor-scheduler"
precision = args.precision

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def tvm_ansor_backend(model: torch.nn.Module, example_inputs):
    # Convert PyTorch model to TVM Relay
    input_name = "input0"
    shape_list = [(input_name, example_inputs.shape)]
    mod, params = tvm.relay.frontend.from_pytorch(model, shape_list)

    # Extract tasks for Ansor
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target="cuda")

    # Set tuning options
    tuning_options = auto_scheduler.TuningOptions(
        num_measure_trials=1000,
        measure_callbacks=[auto_scheduler.RecordToFile("ansor_tuning.json")],
        verbose=1,
    )

    # Run the tuner
    task_scheduler = auto_scheduler.TaskScheduler(tasks, task_weights)
    task_scheduler.tune(tuning_options)

    # Apply the best schedule
    with auto_scheduler.ApplyHistoryBest("ansor_tuning.json"):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = tvm.relay.build(mod, target="cuda", params=params)

    # Create TVM runtime module
    dev = tvm.cuda()
    rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    def compiled_model(*args):
        rt_mod.set_input("input0", tvm.nd.array(args[0].numpy()))
        rt_mod.run()
        return torch.tensor(rt_mod.get_output(0).numpy())

    return compiled_model

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
model = torch.compile(model, mode=mode, backend=tvm_ansor_backend)

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
        output = model.generate(**input_ids, max_new_tokens=256)
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
