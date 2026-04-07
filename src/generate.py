import os
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel

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

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)


def generate_post(input_text):
    """
    입력 문장을 LinkedIn 포스팅으로 변환
    
    Args:
        input_text: 변환할 간단한 문장
    
    Returns:
        LinkedIn 스타일의 생성된 포스팅 텍스트
    """
    prompt = f"Input: {input_text} Output:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 텍스트 생성
    outputs = model.generate(
        inputs["input_ids"],
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # "Output:" 이후 부분만 추출
    if "Output:" in generated_text:
        return generated_text.split("Output:")[1].strip()
    
    return generated_text

# 스크립트 직접 실행 시
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Simple sentence to transform")
    args = parser.parse_args()
    
    post = generate_post(args.input)
    
    print("Generated LinkedIn Post:")
    print(post)