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
subset_length = int(sys.argv[2]) if len(sys.argv) > 2 else 0 # if 0 -> full dataset otherwise subset

print(f"Using model: {model_name}")
print(f"Subset length: {'Full dataset' if subset_length == 0 else subset_length} samples")

# Path to trained model (output_dir must match TrainingArguments)
output_dir = f"/mnt/output/model/{model_name}"

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
texts = df["review"].dropna().astype(str).tolist()

# Apply subset limit if requested
if subset_length > 0:
    texts = texts[:subset_length]

# Create dataset
dataset = Dataset.from_dict({"text": texts})
print(f"Loaded dataset with {len(dataset)} samples.")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=64)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Data collatora
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Training args
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,         # Only overwrites checkpoints
    num_train_epochs=1,                # Start with 1 to reduce runtime and test performance
    per_device_train_batch_size=2,     # Lower batch size to fit in 4GB RAM
    evaluation_strategy="no",          # Skip evaluation to save resources
    save_steps=100,                    # Save less frequently to reduce I/O
    save_total_limit=1,                # Keep only the last checkpoint
    logging_dir="/mnt/output/logs",
    logging_steps=100,                 # Log less often to reduce CPU load
    dataloader_num_workers=0,          # Prevent multithreaded data loading (can cause crashes on low-resource systems)
    disable_tqdm=True                  # Optional: reduce console overhead in slow terminals
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