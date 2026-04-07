import json
import random

"""
LinkedIn 포스팅 스타일의 합성 데이터 생성
간단한 문장을 입력으로 하고, LinkedIn 스타일의 포스팅을 출력으로 하는 데이터셋을 생성합니다.
"""

def generate_linkedin_posts(num_samples=1000):
    """
    LinkedIn 스타일의 입출력 데이터 쌍 생성
    
    Args:
        num_samples: 생성할 데이터 샘플 수 (기본값: 1000)
    
    Returns:
        입출력 쌍의 딕셔너리 리스트
    """
    # 입력 예시들
    inputs = [
        "I learned Python today.",
        "Excited about AI.",
        "Networking tips.",
        "Work-life balance.",
        "New project launch.",
    ]
    
    # 출력 예시들 (LinkedIn 스타일)
    outputs = [
        "Thrilled to announce I've dived into Python! 🚀 Here's my journey: [details]. #Python #Tech",
        "AI is revolutionizing our world! Excited to explore more. 🤖 #AI #Innovation",
        "Networking is key to success. Here are my top tips: [tips]. #Networking #Career",
        "Balancing work and life is essential. My thoughts: [thoughts]. #WorkLifeBalance",
        "Launching a new project today! Can't wait to share updates. 📈 #ProjectLaunch",
    ]

    data = []
    for _ in range(num_samples):
        input_text = random.choice(inputs)
        output_text = random.choice(outputs)
        data.append({"input": input_text, "output": output_text})

    return data

# 스크립트 직접 실행 시
if __name__ == "__main__":
    data = generate_linkedin_posts()
    
    with open("../data/linkedin_posts.json", "w") as f:
        json.dump(data, f, indent=4)
    
    print("Data generated and saved to data/linkedin_posts.json")