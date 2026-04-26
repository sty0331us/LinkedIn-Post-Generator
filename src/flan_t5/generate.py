import os
import unicodedata
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

import transformers
transformers.logging.set_verbosity_error()

from transformers import T5ForConditionalGeneration, T5Tokenizer

_ROOT = Path(__file__).resolve().parents[2]
model_path = str(_ROOT / "models" / "fine_tuned_flan_t5")

if not os.path.isdir(model_path):
    raise FileNotFoundError(
        f"Model not found: {model_path}\nRun src/flan_t5/train.py first."
    )

# 토크나이저 파일이 없으면(학습 중단 등) 베이스 모델에서 로드
tokenizer_source = (
    model_path
    if os.path.isfile(os.path.join(model_path, "tokenizer_config.json"))
    else "google/flan-t5-base"
)
tokenizer = T5Tokenizer.from_pretrained(tokenizer_source)

# 최종 저장본이 없으면 가장 최신 체크포인트에서 모델 로드
def _latest_checkpoint(base_dir):
    checkpoints = [
        d for d in os.listdir(base_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(base_dir, d))
    ]
    if not checkpoints:
        return None
    return os.path.join(base_dir, sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1])

model_source = (
    model_path
    if os.path.isfile(os.path.join(model_path, "config.json"))
    else _latest_checkpoint(model_path)
)
if model_source is None:
    raise FileNotFoundError(f"No model or checkpoint found in {model_path}")
if model_source != model_path:
    print(f"[Info] 체크포인트에서 로드: {os.path.basename(model_source)}")

model = T5ForConditionalGeneration.from_pretrained(model_source)
model.eval()


def is_korean(text):
    return any(unicodedata.name(c, "").startswith("HANGUL") for c in text)


def translate_to_english(text):
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source="ko", target="en").translate(text)


def generate_post(input_text):
    if is_korean(input_text):
        input_text = translate_to_english(input_text)

    prompt = f"Write a LinkedIn post about: {input_text}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)

    with __import__("torch").no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_length=256,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    print("Generated LinkedIn Post:")
    print(generate_post(args.input))
