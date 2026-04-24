import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from generate import generate_post


def generate(input_text):
    if not input_text.strip():
        return "입력 문장을 써주세요."
    return generate_post(input_text)


demo = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(
        label="입력 문장",
        placeholder="예: 오늘 새 프로젝트를 시작했다",
        lines=3
    ),
    outputs=gr.Textbox(
        label="생성된 LinkedIn 포스트",
        lines=6
    ),
    title="LinkedIn 포스트 생성기",
    description="간단한 문장을 LinkedIn 스타일로 변환해드립니다.",
    examples=[
        ["오늘 새 프로젝트를 시작했다"],
        ["I learned Python today."],
        ["Excited about AI."],
    ]
)

if __name__ == "__main__":
    demo.launch()
