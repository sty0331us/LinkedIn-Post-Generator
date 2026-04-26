# LinkedIn Post Generator

간단한 문장을 입력하면 LinkedIn 스타일의 전문적인 포스트로 자동 변환해주는 AI 모델입니다.

> **예시**
> - 입력: `"I quit my job today"`
> - 출력: `"After three years of incredible growth, I've made one of the hardest decisions of my career..."`

---

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [모델 종류](#모델-종류)
3. [프로젝트 구조](#프로젝트-구조)
4. [처음 시작하기 (설치)](#처음-시작하기-설치)
5. [전체 파이프라인](#전체-파이프라인)
6. [웹 UI 실행](#웹-ui-실행)
7. [모델 비교](#모델-비교)
8. [자주 묻는 질문](#자주-묻는-질문)

---

## 프로젝트 개요

이 프로젝트는 두 가지 AI 모델을 파인튜닝(fine-tuning)하여 LinkedIn 포스팅 생성기를 만드는 것입니다.

**파인튜닝이란?**
이미 대량의 텍스트로 학습된 AI 모델에 "LinkedIn 포스트 스타일"의 데이터를 추가로 학습시켜, LinkedIn에 특화된 문체로 글을 쓰도록 만드는 과정입니다.

**전체 흐름:**
```
데이터 준비 → 모델 학습 → 포스트 생성
```

---

## 모델 종류

이 프로젝트에서는 두 가지 모델을 비교합니다:

| 모델 | 특징 | 학습 데이터 |
|------|------|------------|
| **GPT-2** | OpenAI의 텍스트 생성 모델, 자유로운 문체 | 합성 데이터 (Claude/Gemini API로 생성) |
| **Flan-T5** | Google의 지시문 수행 모델, 일관성 우수 | HuggingFace 공개 데이터셋 |

---

## 프로젝트 구조

```
LinkedIn-Post-Generator/
│
├── src/                          # 소스 코드
│   ├── app.py                    # 웹 UI (Gradio)
│   ├── compare.py                # 두 모델 결과 비교
│   │
│   ├── data/                     # 데이터 준비 패키지
│   │   ├── hf.py                 # HuggingFace 데이터 다운로드·전처리
│   │   ├── claude.py             # Claude API로 합성 데이터 생성
│   │   ├── gemini.py             # Gemini API로 합성 데이터 생성
│   │   ├── reformat.py           # 데이터 형식 재가공
│   │   └── explore.py            # 데이터셋 탐색
│   │
│   ├── gpt2/                     # GPT-2 모델 패키지
│   │   ├── train.py              # GPT-2 학습
│   │   └── generate.py           # GPT-2 포스트 생성
│   │
│   └── flan_t5/                  # Flan-T5 모델 패키지
│       ├── train.py              # Flan-T5 학습 (HF 데이터)
│       ├── train_reformatted.py  # Flan-T5 학습 (재가공 데이터)
│       └── generate.py           # Flan-T5 포스트 생성
│
├── data/                         # 학습 데이터 저장 폴더
│   ├── hf_raw.json               # HuggingFace 원본 데이터
│   ├── linkedin_posts_hf.json    # 전처리된 학습 데이터 (HF)
│   └── linkedin_posts_reformatted.json  # 재가공 학습 데이터
│
├── models/                       # 학습된 모델 저장 폴더 (git 제외)
│   ├── fine_tuned_gpt2/          # GPT-2 파인튜닝 결과
│   ├── fine_tuned_flan_t5/       # Flan-T5 파인튜닝 결과 (HF 데이터)
│   └── fine_tuned_flan_t5_new/   # Flan-T5 파인튜닝 결과 (재가공 데이터)
│
├── requirements.txt              # 필요한 패키지 목록
└── README.md                     # 이 파일
```

---

## 처음 시작하기 (설치)

### 1. 저장소 클론
```bash
git clone https://github.com/sty0331us/LinkedIn-Post-Generator.git
cd LinkedIn-Post-Generator
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate       # Windows
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
pip install sentencepiece gradio deep-translator
```

---

## 전체 파이프라인

### Flan-T5 모델 (HuggingFace 데이터 기반) — 권장

#### Step 1. HuggingFace에서 데이터 다운로드
```bash
python -c "
from datasets import load_dataset
import json

ds = load_dataset('ShayanShamsi/prompt_to_linkedin_post', split='train')
data = [{'prompt': row['prompt'], 'output': row['output']} for row in ds]
with open('data/hf_raw.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f'{len(data)}개 저장 완료')
"
```

#### Step 2. 데이터 전처리 (긴 지시문 → 짧은 키워드로 변환)
```bash
python src/data/hf.py
```
> `data/hf_raw.json` → `data/linkedin_posts_hf.json` 으로 변환됩니다.

#### Step 3. 모델 학습
```bash
python src/flan_t5/train.py
```
> Apple Silicon Mac이라면 MPS(GPU)가 자동으로 활성화됩니다.
> 학습 완료 후 `models/fine_tuned_flan_t5/` 에 저장됩니다.

#### Step 4. 포스트 생성 테스트
```bash
python src/flan_t5/generate.py --input "I quit my job today"
python src/flan_t5/generate.py --input "오늘 새 프로젝트를 시작했다"   # 한국어도 가능
```

---

### GPT-2 모델 (Claude/Gemini API 기반)

#### Step 1. API 키 설정
```bash
export ANTHROPIC_API_KEY="your-api-key"   # Claude 사용 시
export GOOGLE_API_KEY="your-api-key"      # Gemini 사용 시
```

#### Step 2. 합성 데이터 생성 및 학습
```bash
# Claude로 데이터 생성 후 학습
python src/gpt2/train.py --generate-data --prepare-with claude

# Gemini로 데이터 생성 후 학습
python src/gpt2/train.py --generate-data --prepare-with gemini

# 기존 데이터로만 학습
python src/gpt2/train.py --data-file data/linkedin_posts_hf.json
```

#### Step 3. 포스트 생성 테스트
```bash
python src/gpt2/generate.py --input "I launched my first product"
```

---

## 웹 UI 실행

모델 학습이 완료된 후 Gradio 기반 웹 인터페이스를 실행할 수 있습니다.

```bash
python src/app.py
```

브라우저에서 `http://127.0.0.1:7860` 접속 → 텍스트 입력 후 생성 버튼 클릭

![웹 UI 예시](https://i.imgur.com/placeholder.png)

> 현재 웹 UI는 Flan-T5 모델을 사용합니다. GPT-2로 변경하려면 `src/app.py`의 import를 수정하세요.

---

## 모델 비교

두 모델의 결과를 나란히 비교하려면:

```bash
python src/compare.py
```

아래와 같이 같은 입력에 대한 세 모델의 출력을 비교합니다:
```
======================================================================
INPUT: starting a new AI project today
----------------------------------------------------------------------
[GPT-2]
Starting a new AI project is always exciting...

[Flan-T5 (HF 데이터)]
Today marks the beginning of something new...

[Flan-T5 (재가공 데이터)]  ← models/fine_tuned_flan_t5_new/ 가 있을 때만 표시
Embarking on a new AI journey...
```

---

## 자주 묻는 질문

**Q. 학습에 GPU가 꼭 필요한가요?**
아니요. CPU로도 학습 가능하지만 매우 느립니다. Apple Silicon Mac이라면 MPS가 자동으로 사용됩니다.

**Q. `models/` 폴더가 GitHub에 올라가지 않아요.**
용량이 크기 때문에 `.gitignore`에 포함되어 있습니다. 각자 로컬에서 학습해야 합니다.

**Q. 한국어 입력은 어떻게 동작하나요?**
`deep-translator` 라이브러리를 통해 Google 번역 API로 영어로 변환 후 모델에 입력합니다.

**Q. 학습이 중간에 끊기면 어떻게 되나요?**
`models/fine_tuned_flan_t5/checkpoint-XXX/` 형태로 중간 저장본이 남습니다. `generate.py`는 최신 체크포인트를 자동으로 불러옵니다.
