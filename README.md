# Optimization of BioGPT-QA for Efficient Biomedical Text Generation

## 🧠 Team Information
- **Team Name**: Optimizing BioGPT
- **Members**:
  - Sanmati Choudhary (sc4789)
  - Wei Qiang (wq2167)
  - Victoria Li (vl2395)

---

## 1. Problem Statement

This project focuses on optimizing BioGPT-Large — a domain-specific transformer language model fine-tuned on the PubMedQA biomedical question answering dataset. The goal is to reduce inference latency and computational cost for real-world deployment in resource-constrained healthcare environments using compiler-level and model compression techniques.

---

## 2. Model Description

- **Model**: BioGPT-Large-PubMedQA (from Hugging Face: `microsoft/biogpt`)
- **Architecture**:
  - 24 transformer decoder layers
  - Causal attention with masked multi-head self-attention
  - 1024 hidden size, 16 attention heads
  - Pretrained on biomedical literature (PubMed abstracts and PMC full texts)
- **Framework**: PyTorch + Hugging Face Transformers
- **Fine-tuning**: The model was fine-tuned on the PubMedQA biomedical question answering dataset to perform yes/no/maybe multiple-choice classification.
- **Modifications**:
    - Integrated quantization and pruning modules into the model architecture for training- and inference-time optimization
    - Enabled torch.compile for dynamic backend selection (e.g., Inductor, AOT Eager, TVM)
    - Unified training and evaluation within single experiment scripts for reproducibility and streamlined benchmarking

---

## 3. Final Results Summary

### Best Combined Configuration

| Metric               | Value                         |
|----------------------|-------------------------------|
| Accuracy             | 54.9%                         |
| Latency              | 2.05 ms/sample                |
| Throughput           | 0.48 samples/sec              |
| GPU Utilization      | 36.23%                        |
| Memory Usage         | 17.92%                        |
| Power Consumption    | 42.81 W                       |
| Temperature          | 44.0°C                        |
| Device               | NVIDIA V100                   |
| Config               | QAT + Structured + Float16 + InductorMax |

---

## 4. Reproducibility Instructions

### A. Requirements
```bash
pip install -r requirements.txt
```

---

### B. WandB Dashboard  
Private project (shared with course staff):  
[wandb.ai/sc4789-columbia-university/biogpt-pubmedqa](https://wandb.ai/sc4789-columbia-university/biogpt-pubmedqa?nw=nwusersc4789)

---

### C. Training and Inference (Unified)

All experiments are run using single scripts that include training/fine-tuning (if applicable) and evaluation. 
We have two pipelines: post-training (precision reduction, tensor optimization, PTQ) and re-training (QAT & pruning)

For example, for tensor optimization experiment, run the following command. Backend arguments include `aot_eager`, `tvm` and `inductor`. Mode arguments are `default`, `ansor`, `max-autotune`.

```bash
# Post-training pipeline (tensor optimization)
python tensor_rewriting/tensor_experiments.py --backend inductor --mode max-autotune
```

For pruning experiment, run the following command. The strategy argument can be changed to `structured` or `block`.
```bash
# Training pipeline (unstructured pruning)
python pruning/pruning_experiments.py --strategy unstructured
```

---

### D. Quickstart: Minimum Reproducible Result
Run the baseline model or a single precision-reduction experiment. Results are shown in appendix for baseline and precision reduction experiment. 

```bash
# Baseline inference (no optimization)
python baseline.py

# Precision reduction (float16)
python precision_reduction/precision_experiments.py --precision float16
```

---

## 5. Notes and Repository Structure

Scripts are organized by technique:

- `baseline.py`: Runs BioGPT without optimization
- `precision_reduction/`: float16, float4, int8, int4 experiments
- `pruning/`: unstructured, structured, block pruning
- `quantization/`: PTQ and QAT
- `tensor_rewriting/`: `torch.compile` backends (`aot_eager`, `tvm`, `inductor`) and modes (`default`, `ansor`, `max-autotune`)
- `combined_experiments/`: Full pipelines for post-training or training-time combinations
- `evaluation/`: PubMed Test & Dev Dataset and ground truth samples that each of our optimized models 

---

## Appendix: Results Tables

### Precision Reduction

| Metric               | Baseline (Float32) | Float16 | Int8 | Float4 | Int4 |
|----------------------|-------------------|---------|------|--------|------|
| Accuracy (%)         | 55.2              | 55.6    | 54.6 | 54.1   | 54.2 |
| Latency (s)          | 1.93              | 1.47    | 6.48 | 3.23   | 3.52 |
| Throughput           | 0.52              | 0.68    | 0.15 | 0.31   | 0.28 |
| GPU Utilization (%)  | 48.06             | 27.96   | 20.42| 21.6   | 21.06|
| Memory Usage (%)     | 34.90             | 16.19   | 2.63 | 5.02   | 4.86 |
| Power Usage (W)      | 59.60             | 56.56   | 40.06| 42.09  | 45.26|
| Temperature (°C)     | 48.65             | 46.64   | 35.95| 36.01  | 35.93|

---

### Pruning

| Metric               | Baseline | Unstructured | Structured | Block Sparsity |
|----------------------|----------|--------------|------------|----------------|
| Accuracy (%)         | 55.2     | 54.2         | 55.2       | 55.2           |
| Latency (s)          | 1.93     | 2.41         | 2.39       | 2.56           |
| Throughput           | 0.52     | 0.418        | 0.42       | 0.38           |
| GPU Utilization (%)  | 48.06    | 41.79        | 58.54      | 54.94          |
| Memory Usage (%)     | 34.90    | 41.26        | 41.87      | 38.99          |
| Power Usage (W)      | 59.60    | 58.85        | 57.78      | 56.86          |
| Temperature (°C)     | 48.65    | 49.93        | 48.54      | 47.98          |

---

### Quantization

| Metric               | Baseline | PTQ        | QAT            |
|----------------------|----------|------------|----------------|
| Accuracy (%)         | 55.2     | 46.6       | 53.2           |
| Latency (s)          | 1.93     | 6.48       | 1.80           |
| Throughput           | 0.52     | 0.20       | 0.545          |
| GPU Utilization (%)  | 48.06    | n/a        | 17.90          |
| Memory Usage (%)     | 34.90    | n/a        | 41.56          | 
| Power Usage (W)      | 59.60    | n/a        | 48.63          |
| Temperature (°C)     | 48.65    | n/a        | 45.34          |

---

### Tensor Optimization

| Metric               | Baseline | AOT Eager | TVM | TVM-Ansor | InductorMax | Inductor |
|----------------------|----------|-----------|-----|-----------|-------------|----------|
| Accuracy (%)         | 55.2     | 55.2      | 55.2| 55.2      | 55.2        | 55.2     |
| Latency (s)          | 2.82     | 1.51      | 2.14| 2.14      | 1.77        | 2.38     |
| Throughput           | 0.35     | 0.49      | 0.47| 0.47      | 0.46        | 0.56     |
| GPU Utilization (%)  | 90.89    | 36.79     | 98  | 97        | 34.40       | 37.99    |
| Memory Usage (%)     | 70       | 26.07     | 43.27| 42.6     | 24.19       | 26.94    |
| Power Usage (W)      | 62.73    | 52.73     | 23.97| 28.26    | 50.39       | 53.6     |
| Temperature (°C)     | 76       | 41.49     | 33  | 48.26     | 40.70       | 41.90    |

---

### Combined Experiments

| Metric               | Float16 + AOT | Float16 + InductorMax | QAT + Structured + AOT | QAT + Structured + InductorMax |
|----------------------|---------------|------------------------|--------------------------|----------------------------------|
| Accuracy (%)         | 55.2          | 55.2                   | 53.8                     | 54.9                             |
| Latency (ms/sample)  | 1.58          | 1.63                   | 2.20                     | 2.05                             |
| Throughput           | 0.62          | 0.61                   | 0.45                     | 0.48                             |
| GPU Utilization (%)  | 26.68         | 24.0                   | 29.0                     | 36.23                            |
| Memory Usage (%)     | 15.44         | 14.87                  | 16.96                    | 17.92                            |
| Power Consumption (W)| 55.18         | 49.26                  | 46.31                    | 42.81                            |
| Temperature (°C)     | 41.68         | 41.40                  | 44.47                    | 44.00                            |

