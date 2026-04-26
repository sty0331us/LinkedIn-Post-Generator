import json
import re
import os
from pathlib import Path

"""
HuggingFace 데이터셋(ShayanShamsi/prompt_to_linkedin_post)을
학습용 input/output 형식으로 변환합니다.

원본: data/hf_raw.json  (prompt가 긴 지시문)
결과: data/linkedin_posts_hf.json  (짧은 input + output)
"""

_ROOT = Path(__file__).resolve().parents[2]

PREFIXES = [
    r"^(generate|compose|create|write|draft|please generate|please write|please create)\s+a\s+linkedin\s+post\s+(about|announcing|expressing|encouraging|that|detailing|regarding|describing|highlighting|sharing|showcasing|discussing|celebrating|congratulating|introducing|covering|summarizing|focusing on|to)?\s*",
    r"^(generate|compose|create|write|draft|please generate|please write|please create)\s+a\s+linkedin\s+post\s+with\s+the\s+following\s+details:\s*[-•]?\s*start with an engaging hook:\s*[\"']?",
]

def shorten_prompt(prompt: str) -> str:
    text = prompt.strip()
    first_line = text.split("\n")[0].strip()

    for pattern in PREFIXES:
        first_line = re.sub(pattern, "", first_line, flags=re.IGNORECASE).strip()

    first_line = first_line.strip('"\'')

    sentences = re.split(r'(?<=[.!?])\s+', first_line)
    result = sentences[0].strip().rstrip(".")

    if len(result) > 60:
        result = result[:60].rsplit(" ", 1)[0]

    return result if result else first_line[:60]


def main():
    input_path = str(_ROOT / "data" / "hf_raw.json")
    output_path = str(_ROOT / "data" / "linkedin_posts_hf.json")

    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    BAD_PATTERNS = ["following details", "engaging hook", "following information", "following points"]

    converted = []
    for item in raw:
        short_input = shorten_prompt(item["prompt"])
        output = item["output"].strip()
        if not short_input or not output:
            continue
        if any(p in short_input.lower() for p in BAD_PATTERNS):
            continue
        if len(short_input) < 5:
            continue
        converted.append({"input": short_input, "output": output})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    print(f"변환 완료: {len(converted)}개 → {output_path}")
    print("\n샘플 5개:")
    for item in converted[:5]:
        print(f"  input : {item['input']}")
        print(f"  output: {item['output'][:80]}...")
        print()


if __name__ == "__main__":
    main()
