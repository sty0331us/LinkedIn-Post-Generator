from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
import json
import torch

"""
GPT-2 모델 파인튜닝
LinkedIn 스타일의 입출력 데이터로 GPT-2 모델을 파인튜닝합니다.
"""

# MPS 디바이스 확인(macOS에서 GPU 사용 가능 여부 확인)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 기본 모델 로드
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 모델을 디바이스로 이동 (macOS에서 MPS 사용 가능 시)
model = model.to(device)

# 훈련 데이터 로드
with open("../data/linkedin_posts.json", "r") as f:
    data = json.load(f)


def preprocess_function(examples):
    """
    데이터를 모델 훈련용 형식으로 변환
    
    Args:
        examples: 입출력 쌍의 리스트
    
    Returns:
        토큰화된 입력과 레이블
    """
    inputs = [f"Input: {ex['input']} Output:" for ex in examples]
    targets = [ex["output"] for ex in examples]
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")["input_ids"]
    model_inputs["labels"] = labels
    
    return model_inputs


# 데이터 전처리
dataset = Dataset.from_list(data)
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 훈련 설정
training_args = TrainingArguments(
    output_dir="../models/fine_tuned_gpt2",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    use_mps_device=True,  # Enable using MPS of Mac if available
)

# 모델 훈련
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# 훈련된 모델 저장
trainer.save_model("../models/fine_tuned_gpt2")
tokenizer.save_pretrained("../models/fine_tuned_gpt2")

print("Model fine-tuned and saved.")