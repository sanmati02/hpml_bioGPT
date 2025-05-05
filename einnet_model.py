import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the pre-trained model (BioGPT in this case)
model_name = "microsoft/BioGPT-Large-PubMedQA"  # Adjust this if needed
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode

# Example input for the model (you can adjust the input depending on your use case)
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_text = "What is the role of bioinformatics in genomics?"
inputs = tokenizer(input_text, return_tensors="pt")

# Export the model to ONNX
onnx_model_path = "biogpt_large_pubmedqa.onnx"  # Path to save the ONNX model
torch.onnx.export(
    model,  # Model to export
    (inputs['input_ids'],),  # Model inputs
    onnx_model_path,  # Where to save the ONNX model
    export_params=True,  # Export model parameters
    opset_version=11,  # ONNX opset version (adjust if necessary)
    input_names=['input_ids'],  # Names of model inputs
    output_names=['output'],  # Names of model outputs
    dynamic_axes={'input_ids': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Dynamic axes for batching
)

print(f"ONNX model saved to {onnx_model_path}")

