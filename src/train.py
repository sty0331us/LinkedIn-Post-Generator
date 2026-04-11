import os
import json
import torch
import socket
from huggingface_hub import configure_http_backend
import requests
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset

# Hugging Face 로그 레벨을 INFO로 설정하여 상세 진행 상황 출력
os.environ["TRANSFORMERS_VERBOSITY"] = "info"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0" # 진행바 강제 활성화

# 무한 대기(Hang) 방지를 위해 기본 네트워크 타임아웃 설정 (300초)
socket.setdefaulttimeout(300)

"""
GPT-2 모델 파인튜닝
LinkedIn 스타일의 입출력 데이터로 GPT-2 모델을 파인튜닝합니다.
"""

# 프로젝트 루트 경로 설정 (실행 위치와 무관하게 안전한 경로 참조)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(script_dir, "..", "data", "linkedin_posts.json"))
output_dir = os.path.abspath(os.path.join(script_dir, "..", "models", "fine_tuned_gpt2"))

# MPS 디바이스 확인(macOS에서 GPU 사용 가능 여부 확인)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[1/6] Using device: {device}")

# 기본 모델 로드
print("[2/6] Downloading/Loading GPT-2 model and tokenizer... (이 작업은 처음에 몇 분 정도 걸릴 수 있습니다)")
model_name = "gpt2"


def backend_factory():
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(max_retries=5)) # 5번 재시도
    return session

configure_http_backend(backend_factory)

try:
    # 토크나이저와 모델을 따로 로드하며 상태 확인
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, local_files_only=False)
    print("  - Tokenizer loaded.")
    
    model = GPT2LMHeadModel.from_pretrained(
        model_name, 
        low_cpu_mem_usage=True,
        # 아래 옵션은 네트워크가 불안정할 때 도움이 됩니다
        resume_download=True      # 이어서 받기 활성화
    )
    print("  - Model weights loaded.")
    
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
except Exception as e:
    print(f"\n❌ 에러 발생: {e}")
    print("현재 네트워크에서 Hugging Face 서버 접속이 원활하지 않습니다.")
    print("팁: 핸드폰 핫스팟(Hotspot)에 연결해서 시도해보거나, VPN을 꺼보세요.")
    exit(1)

# 모델을 디바이스(MPS)로 이동
model = model.to(device)


# 훈련 데이터 로드
print(f"[3/6] Loading training data from {data_path}...")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found at {data_path}")
    
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

def preprocess_function(examples):
    # 입력과 출력을 합쳐서 훈련 텍스트 생성
    texts = [f"Input: {inp} Output: {out}" for inp, out in zip(examples['input'], examples['output'])]
    
    # 토크나이징
    encodings = tokenizer(texts, max_length=512, truncation=True, padding="max_length", return_tensors=None)
    
    # labels는 input_ids와 같음 (causal language modeling)
    encodings["labels"] = encodings["input_ids"].copy()
    
    return encodings

# 데이터 전처리
print("[4/6] Preprocessing and tokenizing dataset...")
dataset = Dataset.from_list(data)
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 훈련 설정
print("[5/6] Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,  # 에폭을 3으로 증가
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    save_steps=100,
    save_total_limit=2,
    use_mps_device=True,
    report_to="none",
    logging_steps=10,
    fp16=False,
    dataloader_num_workers=0
)


# 모델 훈련
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("[6/6] Starting training... (본격적인 학습 시작)")
trainer.train()

# 훈련된 모델 저장
print(f"Saving fine-tuned model to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("Model fine-tuned and saved successfully!")