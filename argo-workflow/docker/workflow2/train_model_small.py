import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling,
)

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

# Dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="data.txt",
    block_size=64
)

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
    train_dataset=dataset,
)

trainer.train()
trainer.save_model(output_dir)