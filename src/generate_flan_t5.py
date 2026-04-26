import os
import unicodedata
import warnings
import logging
warnings.filterwarnings("ignore")

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

import transformers
transformers.logging.set_verbosity_error()

from transformers import T5ForConditionalGeneration, T5Tokenizer

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(script_dir, "..", "models", "fine_tuned_flan_t5"))

if not os.path.isdir(model_path):
    raise FileNotFoundError(
        f"Model not found: {model_path}\nRun src/train_flan_t5.py first."
    )

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
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
