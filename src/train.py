import argparse
import importlib.util
import json
import os
import re
import socket
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import torch
from datasets import Dataset
from huggingface_hub import configure_http_backend
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

"""
GPT-2 모델 파인튜닝
LinkedIn 스타일의 입출력 데이터로 GPT-2 모델을 파인튜닝합니다.
"""

# Hugging Face 로그 레벨을 INFO로 설정하여 상세 진행 상황 출력
os.environ["TRANSFORMERS_VERBOSITY"] = "info"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

# 무한 대기(Hang) 방지를 위해 기본 네트워크 타임아웃 설정 (300초)
socket.setdefaulttimeout(300)

# 프로젝트 루트 경로 설정 (실행 위치와 무관하게 안전한 경로 참조)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = Path(script_dir).parent / "data"
output_dir = Path(script_dir).parent / "models" / "fine_tuned_gpt2"


def get_latest_dataset_file(data_directory: Path) -> Optional[Path]:
    data_directory = Path(data_directory)
    if not data_directory.exists():
        return None

    pattern = re.compile(r"linkedin_posts_(\d{4}_\d{4})\.json$")
    latest_path = None
    latest_timestamp = ""
    for path in data_directory.glob("linkedin_posts_*.json"):
        match = pattern.match(path.name)
        if match:
            timestamp = match.group(1)
            if timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_path = path

    if latest_path is not None:
        return latest_path

    fallback = data_directory / "linkedin_posts.json"
    return fallback if fallback.exists() else None


def load_data_preparation_module(module_name: str):
    module_file = Path(script_dir) / f"data_preparation_{module_name}.py"
    if not module_file.exists():
        raise FileNotFoundError(
            f"Expected module file not found: {module_file}."
        )

    spec = importlib.util.spec_from_file_location(module_file.stem, str(module_file))
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_file}")
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GPT-2 on LinkedIn-style post data."
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to an existing training dataset JSON file.",
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate a new training dataset before training.",
    )
    parser.add_argument(
        "--prepare-with",
        choices=["claude", "gemini"],
        default="claude",
        help="Which data preparation module to use when --generate-data is enabled.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of samples to generate when --generate-data is enabled.",
    )
    return parser.parse_args()


args = parse_args()

# MPS 디바이스 확인(macOS에서 GPU 사용 가능 여부 확인)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[1/6] Using device: {device}")

# 기본 모델 로드
print("[2/6] Downloading/Loading GPT-2 model and tokenizer... (이 작업은 처음에 몇 분 정도 걸릴 수 있습니다)")
model_name = "gpt2"


def backend_factory():
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(max_retries=5))
    return session

configure_http_backend(backend_factory)

try:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, local_files_only=False)
    print("  - Tokenizer loaded.")

    model = GPT2LMHeadModel.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        resume_download=True,
    )
    print("  - Model weights loaded.")

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

except Exception as e:
    print(f"\n Error: {e}")
    print("현재 네트워크에서 Hugging Face 서버 접속이 원활하지 않습니다.")
    print("팁: 핸드폰 핫스팟(Hotspot)에 연결해서 시도해보거나, VPN을 꺼보세요.")
    exit(1)

# 모델을 디바이스로 이동
model = model.to(device)

# 데이터 생성 또는 로드
if args.generate_data:
    if args.data_file and Path(args.data_file).exists():
        raise FileExistsError(
            "The specified --data-file already exists. "
            "Remove the file or omit --generate-data if you want to train on an existing dataset."
        )

    print(f"[3/6] Generating training data using data_preparation_{args.prepare_with}.py...")
    data_prep = load_data_preparation_module(args.prepare_with)
    dataset_path = Path(data_prep.generate_and_save(num_samples=args.num_samples, output_dir=str(data_dir)))
else:
    if args.data_file:
        dataset_path = Path(args.data_file)
    else:
        dataset_path = get_latest_dataset_file(data_dir)

if dataset_path is None or not dataset_path.exists():
    raise FileNotFoundError(
        "Training dataset not found. "
        "Use --generate-data to create one, or pass --data-file <path> to an existing dataset."
    )

print(f"[3/6] Loading training data from {dataset_path}...")
with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)


def preprocess_function(examples):
    texts = [f"Input: {inp} Output: {out}" for inp, out in zip(examples["input"], examples["output"])]
    encodings = tokenizer(texts, max_length=512, truncation=True, padding="max_length", return_tensors=None)
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings


# 데이터 전처리
print("[4/6] Preprocessing and tokenizing dataset...")
dataset = Dataset.from_list(data)
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 훈련 설정
print("[5/6] Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=str(output_dir),
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

# 모델 훈련
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("[6/6] Starting training...")
trainer.train()

# 훈련된 모델 저장
print(f"Saving fine-tuned model to {output_dir}...")
trainer.save_model(str(output_dir))
tokenizer.save_pretrained(str(output_dir))

print("Model fine-tuned and saved successfully!")
