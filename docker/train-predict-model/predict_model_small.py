import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys

# Read model name from command-line argument
model_name = sys.argv[1] if len(sys.argv) > 1 else "distilgpt2"
input_inference = sys.argv[2] if len(sys.argv) > 2 else ""

print(f"Using model: {model_name}")

# Define the path where the model is saved (using the shared volume)
model_path = f"/mnt/output/model/{model_name}"

# Load the model and tokenizer from the saved path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Ensure pad_token_id is set if not defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token

# Example text for inference
input_text = input_inference

# Tokenize input text
inputs = tokenizer(input_text, return_tensors="pt")

# Run inference with no gradients (for evaluation)
with torch.no_grad():
    # Generate text (add max_length and eos_token_id for controlled output)
    outputs = model.generate(
        inputs['input_ids'], 
        max_length=300,  # Limit the length of generated text
        pad_token_id=tokenizer.pad_token_id,  # Ensure padding token is used correctly
        eos_token_id=tokenizer.eos_token_id,  # Stop generation at EOS token
        num_return_sequences=1,  # You can generate multiple outputs if needed
        no_repeat_ngram_size=4,  # Prevent n-gram repetition
        temperature=0.55,  # Set temperature for more diversity (0.7 is a good middle ground)
        top_p=0.80,  # Use nucleus sampling (top-p) to restrict the sample space
        top_k=40,  # Use top-k sampling (optional, can be removed if not needed)
        do_sample=True  # Enable sampling to use temperature and top_p
    )

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Output the result
print(f"Input: {input_text}")
print("\n\n")

# Manually remove the prompt part from the generated text
if generated_text.startswith(input_text):
    generated_text = generated_text[len(input_text):].strip()

print(f"Generated: {generated_text}")
