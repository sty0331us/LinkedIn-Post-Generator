import argparse
import json
import random
from pathlib import Path

import anthropic

"""
LinkedIn 포스팅 스타일의 합성 데이터 생성 (LLM-Assisted + Persona-Based)
Claude API를 사용하여 다양한 페르소나와 주제로 LinkedIn 스타일의 포스팅을 생성합니다.
"""

# --- Technique #5: Persona-Based Generation ---
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

# --- Varied input topics to pair with each persona ---
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


def generate_post(client: anthropic.Anthropic, persona: str, input_text: str) -> str:
    """Call Claude to generate one LinkedIn post for the given persona and input."""
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=300,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},  # cache system prompt across calls
        }],
        messages=[{
            "role": "user",
            "content": f"Persona: {persona}\nInput: {input_text}\n\nWrite the LinkedIn post:",
        }],
    ) as stream:
        return stream.get_final_message().content[0].text.strip()


def generate_linkedin_posts(num_samples: int = 200) -> list[dict]:
    """
    Generate diverse LinkedIn post pairs using Claude (LLM-Assisted + Persona-Based).

    Args:
        num_samples: number of (input, output) pairs to generate

    Returns:
        list of {"input": ..., "output": ...} dicts
    """
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

    # Build all persona × input combinations, shuffle, then take num_samples
    combinations = [(p, i) for p in PERSONAS for i in INPUTS]
    random.shuffle(combinations)
    selected = combinations[:num_samples]

    data = []
    for idx, (persona, input_text) in enumerate(selected, start=1):
        print(f"  [{idx}/{num_samples}] {input_text[:40]!r}  →  {persona[:45]}...")
        output_text = generate_post(client, persona, input_text)
        data.append({"input": input_text, "output": output_text})

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a LinkedIn post dataset using Claude."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of (input, output) pairs to generate (default: 200).",
    )
    args = parser.parse_args()

    print(f"Generating {args.num_samples} samples with {len(PERSONAS)} personas × {len(INPUTS)} topics...")
    print("Make sure ANTHROPIC_API_KEY is set in your environment.\n")

    data = generate_linkedin_posts(num_samples=args.num_samples)

    output_path = Path(__file__).parent.parent / "data" / "linkedin_posts.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\nDone. {len(data)} samples saved to {output_path}")
