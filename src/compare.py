import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEST_INPUTS = [
    "starting a new AI project today",
    "I just got promoted at work",
    "launched a new product with my team",
    "오늘 새로운 것을 배웠다",
]

print("=" * 70)
print("GPT-2 vs Flan-T5 (구 데이터) vs Flan-T5 (새 데이터) 비교")
print("=" * 70)

print("\n[GPT-2 로딩 중...]\n")
from generate import generate_post as gpt2_generate

print("\n[Flan-T5 (구 데이터) 로딩 중...]\n")
from generate_flan_t5 import generate_post as t5_old_generate

print("\n[Flan-T5 (새 데이터) 로딩 중...]\n")

new_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "fine_tuned_flan_t5_new"))
if not os.path.isdir(new_model_path):
    print("⚠️  새 데이터 모델이 아직 학습 중입니다. 두 모델만 비교합니다.\n")
    t5_new_generate = None
else:
    import unicodedata
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    import torch

    _tokenizer = T5Tokenizer.from_pretrained(new_model_path)
    _model = T5ForConditionalGeneration.from_pretrained(new_model_path)
    _model.eval()

    def t5_new_generate(input_text):
        if any(unicodedata.name(c, "").startswith("HANGUL") for c in input_text):
            from deep_translator import GoogleTranslator
            input_text = GoogleTranslator(source="ko", target="en").translate(input_text)
        prompt = f"Write a LinkedIn post about: {input_text}"
        inputs = _tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
        with torch.no_grad():
            output = _model.generate(inputs["input_ids"], max_length=256, num_beams=4, early_stopping=True, no_repeat_ngram_size=3)
        return _tokenizer.decode(output[0], skip_special_tokens=True)

for inp in TEST_INPUTS:
    print("=" * 70)
    print(f"INPUT: {inp}")
    print("-" * 70)
    print("[GPT-2]")
    print(gpt2_generate(inp))
    print()
    print("[Flan-T5 (구 데이터)]")
    print(t5_old_generate(inp))
    print()
    if t5_new_generate:
        print("[Flan-T5 (새 데이터)]")
        print(t5_new_generate(inp))
    print()
