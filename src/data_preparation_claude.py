import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path

import anthropic

"""
CLAUDE 기반 LinkedIn 포스팅 합성 데이터 생성 스크립트
"""

PERSONAS = [
    "a senior software engineer sharing a technical lesson learned on the job",
    "a startup founder announcing a company milestone with humble excitement",
    "a recruiter sharing practical hiring tips and career advice",
    "a recent graduate celebrating landing their first job",
    "a product manager sharing insights about building user-loved products",
    "a data scientist explaining a complex concept in simple terms",
    "a marketing executive sharing a counterintuitive growth strategy",
    "a career coach giving actionable professional development advice",
    "an entrepreneur reflecting honestly on a recent failure and what it taught them",
    "a tech lead discussing team culture and servant leadership",
    "a designer advocating for the importance of UX in product development",
    "a sales professional sharing a lesson from losing a big deal",
]

INPUTS = [
    "I learned Python today.",
    "We just hit 1,000 users.",
    "I got rejected from my dream job.",
    "Remote work changed my perspective.",
    "I shipped my first feature.",
    "Networking feels uncomfortable to me.",
    "I failed a technical interview.",
    "My team launched a new product.",
    "I switched careers at 35.",
    "Open source changed my life.",
    "Work-life balance is harder than I thought.",
    "I mentored a junior developer for the first time.",
    "We closed our seed round.",
    "AI is transforming my industry.",
    "I gave my first conference talk.",
    "I almost quit last year.",
    "My side project got its first paying customer.",
    "I asked for a raise and got it.",
    "A difficult coworker taught me a lot.",
    "I took a week off and it changed everything.",
]

SYSTEM_PROMPT = """You generate LinkedIn posts. Given a persona and a simple input sentence, write a single authentic LinkedIn post in that persona's voice.

Rules:
- 3 to 5 sentences long
- Professional yet personal and specific — no generic filler phrases
- Ends with 1 to 3 relevant hashtags
- No emojis unless they fit naturally
- Return only the post text, nothing else"""


def get_versioned_output_path(output_dir: str, base_name: str = "linkedin_posts") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    return output_dir / f"{base_name}_{timestamp}.json"


def generate_post(client: anthropic.Anthropic, persona: str, input_text: str) -> str:
    """Generate one LinkedIn post using Claude."""
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=300,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"Persona: {persona}\nInput: {input_text}\n\nWrite the LinkedIn post:",
            }
        ],
    ) as stream:
        return stream.get_final_message().content[0].text.strip()


def generate_linkedin_posts(num_samples: int = 200) -> list[dict]:
    client = anthropic.Anthropic()
    combinations = [(p, i) for p in PERSONAS for i in INPUTS]
    random.shuffle(combinations)
    selected = combinations[:num_samples]

    data = []
    for idx, (persona, input_text) in enumerate(selected, start=1):
        print(f"  [{idx}/{num_samples}] {input_text[:40]!r}  ->  {persona[:45]}...")
        output_text = generate_post(client, persona, input_text)
        data.append({"input": input_text, "output": output_text})

    return data


def save_dataset(data: list[dict], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def generate_and_save(num_samples: int = 200, output_dir: str = "data") -> Path:
    data = generate_linkedin_posts(num_samples=num_samples)
    output_path = get_versioned_output_path(output_dir)
    save_dataset(data, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a LinkedIn post dataset using Claude."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of (input, output) pairs to generate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data"),
        help="Directory to save generated dataset files.",
    )
    args = parser.parse_args()

    print(
        f"Generating {args.num_samples} samples with {len(PERSONAS)} personas x {len(INPUTS)} topics..."
    )
    print("Make sure ANTHROPIC_API_KEY is set in your environment.\n")

    output_path = generate_and_save(num_samples=args.num_samples, output_dir=args.output_dir)
    print(f"\nDone. {args.num_samples} samples saved to {output_path}")


if __name__ == "__main__":
    main()
