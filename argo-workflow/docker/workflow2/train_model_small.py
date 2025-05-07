import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# Path to trained model (output_dir must match TrainingArguments)
output_dir = "/mnt/output/model"
model_name = "sshleifer/tiny-gpt2"

# Load tokenizer (same for training or resuming)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Check for existing model
if os.path.exists(output_dir) and os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "pytorch_model.bin")):
    print(f"Loading existing model from {output_dir}")
    model = AutoModelForCausalLM.from_pretrained(output_dir)
else:
    print(f"No existing model found. Starting fresh from {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)

# Load dataset from the local file
dataset = load_dataset('text', data_files={"train": "/app/data.txt"}, split='train')

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
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_steps=10,
    save_total_limit=1,
    logging_dir="/mnt/output/logs",
    logging_steps=5,
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