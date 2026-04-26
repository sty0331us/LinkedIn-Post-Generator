import os
import json
import torch
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset

_ROOT = Path(__file__).resolve().parents[2]
data_path = str(_ROOT / "data" / "linkedin_posts_reformatted.json")
output_dir = str(_ROOT / "models" / "fine_tuned_flan_t5_new")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[1/6] Using device: {device}")

model_name = "google/flan-t5-base"
print(f"[2/6] Loading {model_name}...")
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model = model.to(device)
print("  - Model loaded.")

print(f"[3/6] Loading data from {data_path}...")
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)


def preprocess(examples):
    inputs = [f"Write a LinkedIn post about: {inp}" for inp in examples["input"]]
    targets = examples["output"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = [
        [(t if t != tokenizer.pad_token_id else -100) for t in label]
        for label in labels["input_ids"]
    ]
    return model_inputs


print("[4/6] Preprocessing dataset...")
dataset = Dataset.from_list(data)
tokenized = dataset.map(preprocess, batched=True)

print("[5/6] Setting up training arguments...")
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    save_steps=100,
    save_total_limit=2,
    predict_with_generate=True,
    report_to="none",
    logging_steps=10,
    fp16=False,
    dataloader_num_workers=0,
)

trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=tokenized)

print("[6/6] Starting training...")
trainer.train()

print(f"Saving model to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("Done!")
