"""
규칙 기반으로 LinkedIn 포스트에서 topic/tags를 추출해 input을 재생성합니다.
"""

import json
import re
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "linkedin_posts_hf.json")
output_path = os.path.join(script_dir, "..", "data", "linkedin_posts_reformatted.json")


def extract_first_sentence(text):
    text = text.strip()
    match = re.split(r'(?<=[.!?])\s+', text)
    return match[0].strip() if match else text[:100]


def extract_hashtags(text):
    tags = re.findall(r'#\w+', text)
    return tags[:5]  # 최대 5개


def build_input(post_text):
    topic = extract_first_sentence(post_text)
    tags = extract_hashtags(post_text)

    result = f"topic: {topic}"
    if tags:
        result += f" | tags: {' '.join(tags)}"
    return result


def main():
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"총 {len(data)}개 포스트 처리 중...")

    new_data = []
    for post in data:
        new_input = build_input(post["output"])
        new_data.append({"input": new_input, "output": post["output"]})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"완료! 저장됨: {output_path}")
    print("\n샘플 비교:")
    for i in range(3):
        print(f"\n[{i+1}]")
        print(f"  기존: {data[i]['input']}")
        print(f"  새  : {new_data[i]['input']}")


if __name__ == "__main__":
    main()
