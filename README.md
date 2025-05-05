# BioGPT Optimization

## Project Description:
This project explores inference-time and training-time optimization techniques for BioGPT-Large, a domain-specific transformer model fine-tuned on the PubMedQA biomedical question answering dataset. 

We benchmark and accelerate BioGPT using multiple methods quantization, pruning, tensor rewriting, and precision reduction. Our evaluation metrics include accuracy, throughput, latency, gpu utilization, memory utilization, power and temperature. 

This project is intended to evaluate how well state-of-the-art compiler and quantization techniques can compress, speed up, and deploy large language models like BioGPT in resource-constrained healthcare settings.

## Outline of Code Repository: 
For each of our techniques, we have created a separate folders containing corresponding code files. The following code folders exisit: 

1. Precision Reduction: Contains experiments for float16, float4, int8, int4
2. Pruning: Contains experiments unstructured pruning
3. Quantization: Contains experiments for Quantization Aware Training & Post-Training Quantization 
4. Tensor-rewriting: Contains experiments for different torch.compile modes (`aot_eager`) and backends (`tvm`, `tvm-ansor`, `inductor`, `InductorMax Autotune`, `EiNet`)

The evaluation folder contains the PubMed Test Dataset and ground truth samples that each of our optimized models 

We also include our baseline model code that runs the BioGPT model with no optimization. 

## Example Commands: 

**Precision Reduction**: 
`python precision_reduction/precision_experiments.py --precision int4`
The precision argument can be changed to `float16`, `float4`, or `int8`. 

**Pruning**: 
`python pruning/pruning.py`

**Quantization**: 
`python quantization/quant_int8_dynamic_eval.py`

**TensorRewriting**: 
`python tensor_rewriting/tensor_experiments --backend tvm --mode default`
Backend arguments include `aot_eager`, `tvm` and `inductor`. Mode arguments are `default`, `ansor`, `max-autotune`. 

## Results: 

### Observations

## WandB project link: 

The following is our link to wandb - 
