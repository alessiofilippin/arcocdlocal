import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Define the path where the model is saved (using the shared volume)
model_path = "/mnt/output/model"

# Load the model and tokenizer from the saved path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Example text for inference
input_text = "The quick brown fox jumps over the lazy dog."

# Tokenize input text
inputs = tokenizer(input_text, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model.generate(inputs['input_ids'], max_length=50)

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Output the result
print(f"Input: {input_text}")
print(f"Generated: {generated_text}")
