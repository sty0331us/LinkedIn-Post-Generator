import json
import random
import os

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
    # 입력-출력 매핑 (관련성 있는 쌍)
    data_pairs = [
        # Python 학습 관련
        {
            "input": "I learned Python today.",
            "output": "Thrilled to announce I've dived into Python! 🚀 Here's my journey: mastering new concepts every day. #Python #Tech #LearningJourney"
        },
        {
            "input": "I learned Python today.",
            "output": "Just completed another Python milestone! 💻 The learning never stops. Excited about what I can build with these skills. #Python #Programming"
        },
        {
            "input": "I learned Python today.",
            "output": "Python is becoming my favorite language! 🐍 Today's lesson: building better solutions through clean code. #Python #Developer"
        },
        
        # AI 관련
        {
            "input": "Excited about AI.",
            "output": "AI is revolutionizing our world! 🤖 Excited to explore more possibilities and be part of this transformation. #AI #Innovation #Future"
        },
        {
            "input": "Excited about AI.",
            "output": "The potential of AI is absolutely mind-blowing! 🚀 Can't wait to see how it transforms industries. #AI #MachineLearning #FutureOfWork"
        },
        {
            "input": "Excited about AI.",
            "output": "Every breakthrough in AI brings us closer to solving real-world problems! 🧠 This is truly an exciting time to be part of it. #AI #Technology"
        },
        
        # 네트워킹 관련
        {
            "input": "Networking tips.",
            "output": "Networking is key to success! 🤝 Here are my top tips: be genuine, follow up, and always add value. #Networking #Career #Growth"
        },
        {
            "input": "Networking tips.",
            "output": "Your network is your net worth! Building authentic relationships opens doors. Here's what I've learned: show interest, give first. #Networking"
        },
        {
            "input": "Networking tips.",
            "output": "Pro tip: Quality over quantity in networking. One meaningful connection beats 100 shallow ones. #Networking #CareerAdvice #Professional"
        },
        
        # 일과 삶의 균형
        {
            "input": "Work-life balance.",
            "output": "Balancing work and life is essential! 🌟 My thoughts: Take time for yourself, your health matters. #WorkLifeBalance #MentalHealth #Wellness"
        },
        {
            "input": "Work-life balance.",
            "output": "Remember: It's not about working harder, it's about working smarter and taking care of yourself. 💪 #WorkLifeBalance #Productivity"
        },
        {
            "input": "Work-life balance.",
            "output": "Taking care of your well-being isn't selfish—it's essential! Balance leads to better productivity and happiness. #WorkLifeBalance #SelfCare"
        },
        
        # 프로젝트 런칭
        {
            "input": "New project launch.",
            "output": "Launching a new project today! 🎉 Can't wait to share updates with you all. #ProjectLaunch #Excited #NewBeginnings"
        },
        {
            "input": "New project launch.",
            "output": "From concept to launch—what a journey! 📈 Grateful for the amazing team and excited about the impact. #ProjectLaunch #Teamwork"
        },
        {
            "input": "New project launch.",
            "output": "So proud to announce the launch of our new project! 🚀 This represents months of dedication and innovation. #ProjectLaunch #Achievement"
        },
        
        # 승진 축하
        {
            "input": "I am excited to be promoted.",
            "output": "Thrilled to announce my promotion! 🎊 Grateful for the opportunity and the amazing team's support. This is just the beginning! #CareerGrowth #NewChapter"
        },
        {
            "input": "I am excited to be promoted.",
            "output": "Excited to take on this new role! 📈 Can't thank everyone enough who believed in me. Let's make great things happen together! #Promotion #Leadership"
        },
        {
            "input": "I am excited to be promoted.",
            "output": "What a proud moment! 🌟 Promoted to a new position and ready to bring my A-game. Here's to new challenges and growth! #NewRole #CareerMilestone"
        },
        
        # 실직 관련
        {
            "input": "I am fired.",
            "output": "Today marks the end of one chapter and the beginning of another. 📖 Grateful for the experience and lessons learned. Excited for what's next! #NewBeginning"
        },
        {
            "input": "I am fired.",
            "output": "Every ending is a new opportunity! 🌱 While this wasn't the plan, I'm seeing it as a chance to pursue new possibilities. #Resilience #Growth"
        },
        {
            "input": "I am fired.",
            "output": "Sometimes change comes when we least expect it. 💪 But I'm choosing to see this as motivation to find something even better aligned with my goals. #Positive #NextChapter"
        },
        
        # 구직/오픈 상태
        {
            "input": "I am open status now.",
            "output": "Exciting news! 🎯 I'm now open to new opportunities. If you know of any roles that might be a great fit, let's connect! #Hiring #OpenToOpportunities"
        },
        {
            "input": "I am open status now.",
            "output": "I'm officially on the market! 💼 Looking for my next challenge in a role where I can make a real impact. Connections and referrals welcome! #NowHiring #LookingForWork"
        },
        {
            "input": "I am open status now.",
            "output": "Ready for the next adventure! 🚀 Open to exploring new opportunities that align with my skills and passion. Let's talk! #OpenToWork #NewOpportunity"
        },
    ]
    
    # 데이터 쌍을 복제하여 num_samples 개 만들기
    data = []
    for i in range(num_samples):
        pair = data_pairs[i % len(data_pairs)]
        data.append(pair)
    
    # 데이터 셔플
    random.shuffle(data)
    
    return data

# 스크립트 직접 실행 시
if __name__ == "__main__":
    data = generate_linkedin_posts()
    
    # Get the absolute path to the data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, "data")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Save data to file
    output_file = os.path.join(data_dir, "linkedin_posts.json")
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"Data generated and saved to {output_file}")