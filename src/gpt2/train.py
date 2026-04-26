import argparse
import importlib.util
import json
import os
import re
import socket
from pathlib import Path
from typing import Optional

import requests
import torch
from datasets import Dataset
from huggingface_hub import configure_http_backend
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

_ROOT = Path(__file__).resolve().parents[2]
data_dir = _ROOT / "data"
data_path = str(data_dir / "linkedin_posts_hf.json")
output_dir = str(_ROOT / "models" / "fine_tuned_gpt2")

os.environ["TRANSFORMERS_VERBOSITY"] = "info"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
socket.setdefaulttimeout(300)


def get_latest_dataset_file(directory: Path) -> Optional[Path]:
    if not directory.exists():
        return None
    pattern = re.compile(r"linkedin_posts_(\d{4}_\d{4})\.json$")
    latest_path, latest_timestamp = None, ""
    for path in directory.glob("linkedin_posts_*.json"):
        match = pattern.match(path.name)
        if match:
            timestamp = match.group(1)
            if timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_path = path
    if latest_path:
        return latest_path
    fallback = directory / "linkedin_posts.json"
    return fallback if fallback.exists() else None


def load_data_preparation_module(module_name: str):
    module_file = _ROOT / "src" / "data" / f"{module_name}.py"
    if not module_file.exists():
        raise FileNotFoundError(f"Module not found: {module_file}")
    spec = importlib.util.spec_from_file_location(module_file.stem, str(module_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT-2 on LinkedIn-style post data.")
    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--generate-data", action="store_true")
    parser.add_argument("--prepare-with", choices=["claude", "gemini"], default="claude")
    parser.add_argument("--num-samples", type=int, default=200)
    return parser.parse_args()


args = parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[1/6] Using device: {device}")

print("[2/6] Loading GPT-2 model and tokenizer...")
model_name = "gpt2"


def backend_factory():
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(max_retries=5))
    return session


configure_http_backend(backend_factory)

try:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, local_files_only=False)
    model = GPT2LMHeadModel.from_pretrained(model_name, low_cpu_mem_usage=True, resume_download=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    print("  - Model loaded.")
except Exception as e:
    print(f"\nError: {e}")
    exit(1)

model = model.to(device)

if args.generate_data:
    print(f"[3/6] Generating training data using data/{args.prepare_with}.py...")
    data_prep = load_data_preparation_module(args.prepare_with)
    dataset_path = Path(data_prep.generate_and_save(num_samples=args.num_samples))
else:
    dataset_path = Path(args.data_file) if args.data_file else get_latest_dataset_file(data_dir)

if dataset_path is None or not dataset_path.exists():
    raise FileNotFoundError(
        "Training dataset not found. "
        "Use --generate-data to create one, or pass --data-file <path>."
    )

print(f"[3/6] Loading training data from {dataset_path}...")
with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)


def preprocess_function(examples):
    texts = [f"Input: {inp} Output: {out}" for inp, out in zip(examples["input"], examples["output"])]
    encodings = tokenizer(texts, max_length=512, truncation=True, padding="max_length", return_tensors=None)
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings


print("[4/6] Preprocessing dataset...")
dataset = Dataset.from_list(data)
tokenized_dataset = dataset.map(preprocess_function, batched=True)

print("[5/6] Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    save_steps=100,
    save_total_limit=2,
    report_to="none",
    logging_steps=10,
    fp16=False,
    dataloader_num_workers=0,
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)

print("[6/6] Starting training...")
trainer.train()

print(f"Saving model to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("Done!")
