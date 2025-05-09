import os
import sys
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

# Read model name from command-line argument
model_name = sys.argv[1] if len(sys.argv) > 1 else "distilgpt2"

print(f"Using model: {model_name}")

# Path to trained model (output_dir must match TrainingArguments)
output_dir = "/mnt/output/model/{model_name}"

# Load tokenizer (same for training or resuming)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the padding token to be the EOS token
tokenizer.pad_token = tokenizer.eos_token

# Check for existing model in either safetensors or PyTorch binary format
safetensors_path = os.path.join(output_dir, "model.safetensors")
pytorch_bin_path = os.path.join(output_dir, "pytorch_model.bin")

if os.path.exists(output_dir) and os.path.isdir(output_dir) and (os.path.exists(safetensors_path) or os.path.exists(pytorch_bin_path)):
    print(f"Loading existing model from {output_dir}")
    model = AutoModelForCausalLM.from_pretrained(output_dir)
else:
    print(f"No existing model found. Starting fresh from {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)

# Load and process IMDB dataset
csv_path = "/app/IMDB_Dataset.csv"
df = pd.read_csv(csv_path)

# Optionally, filter or clean data
texts = df["review"].dropna().astype(str).tolist()

# Create a custom dataset
dataset = Dataset.from_dict({"text": texts})

# Verify that the dataset was loaded correctly
print(f"Loaded dataset with {len(dataset)} samples.")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=64)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Training args
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,  # overwrite only means checkpoint, not starting from scratch
    num_train_epochs=2,
    per_device_train_batch_size=2,
    save_steps=10,
    save_total_limit=1,
    logging_dir="/mnt/output/logs",
    logging_steps=5,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,
)

trainer.train()

# Save the model and tokenizer
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)  # Save model weights
print("Files in output directory:", os.listdir(output_dir))