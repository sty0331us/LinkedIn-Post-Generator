import os
import argparse
import torch
import warnings
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 경고 메시지 억제
warnings.filterwarnings("ignore", message=".*attention_mask.*")
warnings.filterwarnings("ignore", message=".*pad_token_id.*")

"""
LinkedIn 포스팅 생성 (추론)
파인튜닝된 GPT-2 모델을 사용하여 간단한 문장을 LinkedIn 스타일로 변환합니다.
"""

# 파인튜닝된 모델 로드
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(script_dir, "..", "models", "fine_tuned_gpt2"))

if not os.path.isdir(model_path):
    raise FileNotFoundError(
        f"Model directory not found: {model_path}\n"
        "Please make sure the fine-tuned model was saved there, or run src/train.py first."
    )

# 원본 gpt2 토크나이저 사용 (저장된 토크나이저 대신)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 파인튜닝된 모델 로드
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()
model.config.pad_token_id = tokenizer.eos_token_id


def generate_post(input_text):
    """
    입력 문장을 LinkedIn 포스팅으로 변환
    
    Args:
        input_text: 변환할 간단한 문장
    
    Returns:
        LinkedIn 스타일의 생성된 포스팅 텍스트
    """
    prompt = f"Input: {input_text} Output:"
    
    # 텍스트 인코딩 (attention_mask 포함)
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
    
    # 텍스트 생성
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=120,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # "Output:" 이후 부분만 추출
    if "Output:" in generated_text:
        result = generated_text.split("Output:")[1].strip()
        return result if result else generated_text
    
    return generated_text


# 스크립트 직접 실행 시
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Simple sentence to transform")
    args = parser.parse_args()
    
    post = generate_post(args.input)
    
    print("Generated LinkedIn Post:")
    print(post)