"""
tensor_experiments.py

Runs BioGPT-Large-PubMedQA different compile and backend methods
and logs inference metrics such as accuracy, latency, GPU stats using Weights & Biases.
"""


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
import subprocess

# Argument Parser for Backend + Mode #
parser = argparse.ArgumentParser()
parser.add_argument("--backend", type=str, default="inductor", help="Backend to use: tvm, inductor, etc.")
parser.add_argument("--mode", type=str, default="default", help="Compile mode (default, max-autotune, etc.)")
parser.add_argument("--precision", type=str, default="float32", help="Precision level")
args = parser.parse_args()

backend = args.backend
mode = args.mode
precision = args.precision

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Initialize WandB
wandb.init(
    project="biogpt-pubmedqa",
    name=f"compile-{backend}-{mode}-{precision}",
    config={"backend": backend, "mode": mode, "precision": precision, "task": "QA-PubMedQA"}
)

# Load tokenizer and model 
tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large-PubMEDQA")
model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large-PubMEDQA").cuda()
model.eval()

# Backend Options
if backend == "inductor":
    model = torch.compile(model, backend="inductor", mode=mode)
elif backend == "aot_eager":
    model = torch.compile(model, backend="aot_eager", mode=mode)
elif backend == "tvm":
    if mode != "ansor":
        model = torch.compile(model, backend="tvm", mode=mode)
    else:
        from tvm import auto_scheduler, relay
        import tvm
        def tvm_backend(model, example_inputs):
            input_name = "input0"
            shape_list = [(input_name, example_inputs.shape)]
            mod, params = relay.frontend.from_pytorch(model, shape_list)

            if mode == "ansor":
                tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target="cuda")
                tuning_options = auto_scheduler.TuningOptions(
                    num_measure_trials=1000,
                    measure_callbacks=[auto_scheduler.RecordToFile("ansor_tuning.json")],
                    verbose=1,
                )
                task_scheduler = auto_scheduler.TaskScheduler(tasks, task_weights)
                task_scheduler.tune(tuning_options)
                with auto_scheduler.ApplyHistoryBest("ansor_tuning.json"):
                    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                        lib = relay.build(mod, target="cuda", params=params)
            else:
                with tvm.transform.PassContext(opt_level=3):
                    lib = relay.build(mod, target="cuda", params=params)

            dev = tvm.cuda()
            rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

            def compiled_model(*args):
                rt_mod.set_input("input0", tvm.nd.array(args[0].cpu().numpy()))
                rt_mod.run()
                return torch.tensor(rt_mod.get_output(0).numpy())

            return compiled_model

        example_input = torch.randint(0, 100, (1, 256), dtype=torch.int64).cuda()
        model = tvm_backend(model, example_input)

# Load test data
with open("../evaluation/test_set.json", "r") as f:
    test_data = json.load(f)
with open("../evaluation/test_ground_truth.json", "r") as f:
    test_gt = json.load(f)

# Join QA with labels
data = [{"question": item["QUESTION"], "label": test_gt[pmid]} for pmid, item in test_data.items()]
test_dataset = Dataset.from_list(data)

# Tokenizes the input question and context into a prompt format suitable for generation.
def preprocess(example):
    prompt = example["question"].strip()
    if not prompt.endswith("."): prompt += "."
    full_prompt = prompt + " Answer in the following format in yes or no. the answer to the question given the context is"
    return tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)

#  Decodes generated tokens and extracts the predicted answer from the output string.
def extract_answer(output):
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    if "yes" in decoded.lower(): return "yes"
    if "no" in decoded.lower(): return "no"
    if "maybe" in decoded.lower(): return "maybe"
    return "failed"

# Preprocess inputs and prepare label list
inputs = [preprocess(ex) for ex in test_dataset]
y_true = [ex["label"] for ex in test_dataset]
y_pred, latencies, gpu_util, mem_util, power_usage, temps = [], [], [], [], [], []

# Queries GPU statistics including utilization, memory, power, and temperature.
def get_gpu_info():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,power.draw,temperature.gpu', '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    gpu_info = result.stdout.decode('utf-8').strip().split(',')
    return {
        'gpu_util': float(gpu_info[0]),
        'mem_util': float(gpu_info[1]),
        'power': float(gpu_info[2]),
        'temp': float(gpu_info[3])
    }

# Loop through each preprocessed input and perform inference
for input_ids in tqdm(inputs):
    # Move inputs to GPU
    input_ids = {k: v.to("cuda") for k, v in input_ids.items()}
    # Sync GPU before and after inference for accurate latency measurement
    torch.cuda.synchronize()
    # Log inference time
    start = time.time()
    with torch.no_grad():
        # Generate model output 
        output = model.generate(**input_ids, max_new_tokens=256)
    torch.cuda.synchronize()
    latencies.append(time.time() - start)
    # Decode model output and append to predictions
    y_pred.append(extract_answer(output))

    # Collect and store GPU stats for each inference step
    gpu = get_gpu_info()
    gpu_util.append(gpu['gpu_util'])
    mem_util.append(gpu['mem_util'])
    power_usage.append(gpu['power'])
    temps.append(gpu['temp'])


# Compute final metrics
accuracy = accuracy_score(y_true, y_pred)
thrpt = len(y_pred) / sum(latencies)

# Log GPU stats
wandb.log({
    "accuracy": accuracy,
    "throughput": thrpt,
    "avg_latency": sum(latencies)/len(latencies),
    "avg_gpu_utilization": sum(gpu_util)/len(gpu_util),
    "avg_memory_utilization": sum(mem_util)/len(mem_util),
    "avg_power_usage": sum(power_usage)/len(power_usage),
    "avg_temperature": sum(temps)/len(temps),
})

wandb.finish()
print(f"✅ Accuracy: {accuracy:.4f}, Throughput: {thrpt:.2f} samples/sec")
print(f"Avg GPU Util: {sum(gpu_util)/len(gpu_util):.2f}%")
print(f"Avg Mem Util: {sum(mem_util)/len(mem_util):.2f}%")
print(f"Avg Power: {sum(power_usage)/len(power_usage):.2f} W")
print(f"Avg Temp: {sum(temps)/len(temps):.2f} °C")

