import os
import argparse
import torch
import warnings
import logging
import unicodedata
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

import transformers
transformers.logging.set_verbosity_error()

from transformers import GPT2Tokenizer, GPT2LMHeadModel

_ROOT = Path(__file__).resolve().parents[2]
model_path = str(_ROOT / "models" / "fine_tuned_gpt2")

if not os.path.isdir(model_path):
    raise FileNotFoundError(
        f"Model directory not found: {model_path}\n"
        "Please run src/gpt2/train.py first."
    )

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()
model.config.pad_token_id = tokenizer.eos_token_id


def is_korean(text):
    return any(unicodedata.name(c, "").startswith("HANGUL") for c in text)


def translate_to_english(text):
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source="ko", target="en").translate(text)


def generate_post(input_text):
    if is_korean(input_text):
        input_text = translate_to_english(input_text)

    prompt = f"Input: {input_text} Output:"
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True)

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    if "Output:" in generated_text:
        result = generated_text.split("Output:")[1].strip()
        return result if result else generated_text

    return generated_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    print("Generated LinkedIn Post:")
    print(generate_post(args.input))
