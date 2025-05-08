# BioGPT Optimization

## Project Description:
This project explores inference-time and training-time optimization techniques for BioGPT-Large, a domain-specific transformer model fine-tuned on the PubMedQA biomedical question answering dataset. 

We benchmark and accelerate BioGPT using multiple methods quantization, pruning, tensor rewriting, and precision reduction. Our evaluation metrics include accuracy, throughput, latency, gpu utilization, memory utilization, power and temperature. 

This project is intended to evaluate how well state-of-the-art compiler and quantization techniques can compress, speed up, and deploy large language models like BioGPT in resource-constrained healthcare settings.

## Outline of Code Repository: 
For each of our techniques, we have created a separate folders containing corresponding code files. The following code folders exisit: 

1. Precision Reduction: Contains experiments for float16, float4, int8, int4
2. Pruning: Contains experiments such as unstructured, structured, and block pruning
3. Quantization: Contains experiments for Quantization Aware Training & Post-Training Quantization 
4. Tensor-rewriting: Contains experiments for different torch.compile modes (`aot_eager`) and backends (`tvm`, `tvm-ansor`, `inductor`, `InductorMax Autotune`)
5. Combined Experiments: Contains experiments for running our best methods across post-training (float16, inductor/aot_eager) and during training (float16, inductor/aot_eager, QAT, structured pruning). 

The evaluation folder contains the PubMed Test Dataset and ground truth samples that each of our optimized models 

We also include our baseline model code that runs the BioGPT model with no optimization. 

## Example Commands: 

**Precision Reduction**: 
`python precision_reduction/precision_experiments.py --precision int4`
The precision argument can be changed to `float16`, `float4`, or `int8`. 

**Pruning**: 
`python pruning/pruning_experiments.py --strategy unstructured`
The strategy argument can be changed to `structured`, or `block` 

**Quantization**: 
`python quantization/ptq_dynamic_int8.py`

**Tensor Optimization**: 
`python tensor_rewriting/tensor_experiments.py --backend tvm --mode default`
Backend arguments include `aot_eager`, `tvm` and `inductor`. Mode arguments are `default`, `ansor`, `max-autotune`. 

**Combined Experiments**: 
`python combined_experiments/during_training.py --backend inductor`

`python combined_experiments/post_training_combined.py --backend inductor`

Backend arguments include `aot_eager` and `inductor`. 



## Results: 

**Precision Reduction**: 

<img width="853" alt="image" src="https://github.com/user-attachments/assets/e503683b-7118-47e0-98d0-8cf396aee238" />

- Float16 offers the best trade-off: it matches or slightly improves accuracy over baseline with lowest latency and highest throughput
- Int8, Float4, and Int4 all retain reasonable accuracy but suffer from significantly higher latency and lower throughput
- GPU/Memory utilization sharply drops with lower precisions, showing potential for resource-constrained environment



**Pruning**: 

<img width="852" alt="image" src="https://github.com/user-attachments/assets/cb4da44d-1e6f-4935-a0b7-513b0e3fae88" />

- Structured pruning preserved accuracy and maximized GPU utilization, making it the most deployment-friendly strategy.
- Unstructured pruning slightly dropped accuracy and hurt inference speed, showing poor GPU execution efficiency.
- Block sparsity pruning led to the worst latency (2.56s) and lowest throughput, despite maintaining accuracy — likely due to memory and kernel overhead.


  
**Quantization**: 

<img width="536" alt="image" src="https://github.com/user-attachments/assets/f72733f4-fad4-4eca-9b25-041dbcd7a56c" />

- QAT significantly improves efficiency over baseline with reduced latency and higher throughput while maintaining competitive accuracy
- PTQ results in a large accuracy drop and much slower inference 



**Tensor Optimization**: 

<img width="1091" alt="image" src="https://github.com/user-attachments/assets/e1ec77f6-ec73-4b07-a49f-88b6cc9ceb8a" /> 

- Accuracy stays constant at 55.2% across all tensor rewrite methods
- AOT_Eager and InductorMaxAutotune offer the best trade-off with reduced latency and balanced GPU/power efficiency
- TVM and TVM_Ansor show unusually high GPU utilization but yield longer latency and lower throughput

**Combined Experiments**: 

<img width="749" alt="image" src="https://github.com/user-attachments/assets/1949ffca-e460-493d-a00e-c466fc4ba4db" />

- Best all-around trade-off: Combining QAT + structured pruning + float16 + InductorMax achieved strong accuracy with improved latency, throughput, and hardware efficiency
- Fastest configuration: Float16 + InductorMax reached the lowest latency (1.47s/sample) and highest throughput (0.68 samples/sec), making it ideal when speed is the priority
- Stacking orthogonal techniques (weight pruning, quantization, compiler optimization) enables practical deployment of BioGPT without major performance loss


## WandB project link: 

The following is our link to wandb. While the project is private, it has been shared with the teaching staff of the course. 

https://wandb.ai/sc4789-columbia-university/biogpt-pubmedqa?nw=nwusersc4789. 
