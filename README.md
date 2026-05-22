# LLM 기반 아동 동화 자동 생성 및 안전성 평가 시스템

## 폴더 구조

```
fairytale_llm/
├── src/
│   ├── __init__.py
│   ├── generator.py      # 카나나 nano 2.1B 동화 생성
│   ├── safeguard.py      # 카나나 세이프가드 8B 1차 평가
│   ├── evaluator.py      # Solar API 2차 평가 + Prompt Rewriter
│   ├── pipeline.py       # 전체 파이프라인 오케스트레이터
│   └── data_loader.py    # 동화 데이터셋 로더
├── data/                 # 아동 동화 JSON 데이터셋 (399개)
├── outputs/              # 생성 결과 저장 (자동 생성)
├── main.py               # CLI 진입점
├── .env                  # API 키 설정
├── .env.example          # 환경변수 예시
└── requirements.txt
```
<img width="820" height="700" alt="Image" src="https://github.com/user-attachments/assets/e0e51a90-b53b-45f1-820f-dafc8d7eeefd" />
---

## 설치

```bash
# 1. 가상환경 활성화
conda activate fairytale_llm

# 2. 의존성 설치
pip install -r requirements.txt

# 3. .env 파일 설정
cp .env.example .env
# .env 파일에 UPSTAGE_API_KEY 입력
```

---

## 실행

### 대화형 모드 (권장)
```bash
python main.py
```

### 단일 요청
```bash
python main.py --request "아이가 나비를 찢어죽였어. 그러면 안된다는 교훈을 주는 동화를 써줘."
```

### 데이터셋 통계 확인
```bash
python main.py --stats
```

### 빠른 테스트 (세이프가드 없이)
```bash
python main.py --no-safeguard --request "친구와 사이좋게 지내는 동화를 써줘."
```

---

## 파이프라인 흐름

```
사용자 입력
  ↓
[편향 키워드 검사] ← 왕자·공주·아들·딸 등 성별/인종 편향 단어 거부
  ↓
[1단계] 카나나 nano 2.1B → 동화 생성
  ↓
[길이 검사] ← 6~7세 기준: 550~850자 / 130~210어절
  ↓
[2단계 1차] 카나나 세이프가드 8B → 문장 단위 SAFE/UNSAFE 태깅
  ↓
[2단계 2차] Solar API → CSM 기준 4항목 평가 (5점 만점, 4.5점 이상 합격)
  ├─ PASS → 최종 출력
  └─ FAIL → Prompt Rewriter → 재생성 (최대 3회)
```

---

## 평가 기준 (Common Sense Media 기반)

| 항목 | 설명 | 배점 |
|------|------|------|
| 서사적 맥락 | 갈등→반성→화해 구조가 완결되는가 | 1~5점 |
| 아동 모델링 | 따라하고 싶은 긍정적 행동이 모델링되는가 | 1~5점 |
| 도덕 메시지 | 교훈이 자연스럽게 전달되는가 | 1~5점 |
| 편견·고정관념 | 성역할·인종 편견이 없는가 | 1~5점 |

**합격 기준: 평균 4.5점 이상**

---

## 동화 길이 기준 (근거)

- Valentini et al. (2023): 유아 동화 평균 100~200 어절
- 한국어린이문학협의회: 5~7세 그림책 평균 500~900자
- **본 시스템 기준: 550~850자 / 130~210어절 (6~7세)**

---

## 편향 방지 키워드 목록

다음 단어가 포함된 요청은 거부됩니다:
`왕자`, `공주`, `아들`, `딸`, `남자아이`, `여자아이`, `소년`, `소녀`,
`남자는`, `여자는`, `흑인`, `백인`, `동양인`, `서양인`

---

## VRAM 사용량 (참고)

| 모델 | 파라미터 | 예상 VRAM (fp16) |
|------|----------|-----------------|
| 카나나 nano 2.1B | 2.1B | ~5GB |
| 카나나 세이프가드 8B | 8B | ~16GB |
| 합계 | ~10B | ~21GB |

A100 80GB 환경에서 여유 있게 동시 운용 가능.
단, VRAM 절약을 위해 Generator → 언로드 → Safeguard 순으로 로드하는
sequential 모드 옵션도 pipeline.py에서 조정 가능.
