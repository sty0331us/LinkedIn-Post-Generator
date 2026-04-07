# LinkedIn Post Generator

## 프로젝트 목표
이 프로젝트는 간단한 문장이나 단어들을 입력받아 LinkedIn 포스팅 스타일로 변환하는 LLM (Large Language Model)을 개발하는 것을 목표로 합니다. 사용자가 제공한 간단한 텍스트를 전문적이고 매력적인 LinkedIn 포스트로 재구성하여, 네트워킹과 콘텐츠 공유를 돕습니다.

## 사용되는 모델
- **모델**: GPT-2 (Hugging Face의 `gpt2` 모델)
  - 이유: 텍스트 생성에 특화된 오픈소스 모델로, 파인튜닝이 쉽고, LinkedIn 스타일의 창의적 텍스트 생성에 적합합니다. 더 큰 모델(Llama 2 등)을 사용할 수도 있지만, 시작점으로 GPT-2를 선택했습니다.

## 데이터 종류
- **데이터**: 합성 데이터셋
  - 입력: 간단한 문장이나 단어들 (예: "I learned Python today.")
  - 출력: LinkedIn 스타일 포스팅 (예: "Excited to share that I've mastered Python! 🚀 Here's what I learned: [details]. #Python #Coding")
  - 데이터 생성: `src/data_preparation.py`에서 합성 데이터를 생성합니다. 실제 LinkedIn 데이터를 수집할 수 없으므로, 템플릿 기반으로 생성.

## 프로젝트 구조
```
LinkedIn-Post-Generator/
├── src/
│   ├── data_preparation.py  # 데이터 생성 스크립트
│   ├── train.py             # 모델 파인튜닝 스크립트
│   └── generate.py          # 추론 스크립트
├── data/
│   └── linkedin_posts.json  # 생성된 데이터셋
├── models/
│   └── fine_tuned_gpt2/     # 파인튜닝된 모델 저장 디렉토리
├── requirements.txt          # 의존성 파일
└── README.md                 # 이 파일
```

## 설치 및 실행
1. 의존성 설치: `pip install -r requirements.txt`
2. 데이터 준비: `python src/data_preparation.py`
3. 모델 파인튜닝: `python src/train.py`
4. 포스트 생성: `python src/generate.py --input "Your simple sentence"`

## 참고
- 모델은 Hugging Face에서 다운로드되며, 파인튜닝을 위해 GPU가 권장됩니다.
- 데이터는 합성이므로, 실제 성능 향상을 위해 더 많은 실제 데이터를 추가하세요.