# LLM 기반 아동 동화 자동 생성 및 안전성 평가 시스템

> Kakao Corp.의 **Kanana** 모델과 Upstage의 **Solar Pro 3**를 활용한 아동 동화 자동 생성 및 2단계 안전성 검증 파이프라인

---

## 폴더 구조

```
fairytale_llm/
├── src/
│   ├── __init__.py
│   ├── generator.py        # kanana-1.5-8b 동화 생성
│   ├── safeguard.py        # kanana-safeguard-8b 1차 평가 (S5 LoRA 파인튜닝 포함)
│   ├── evaluator.py        # Solar Pro 3 — 동화 기획 + 2차 평가 + 수정 지시 생성
│   ├── pipeline.py         # 전체 파이프라인 오케스트레이터
│   └── data_loader.py      # 동화 데이터셋 로더 (퓨샷 고정 사용)
├── finetune/
│   └── kanana-s5-adapter/  # S5(그루밍) 탐지 LoRA 어댑터
├── data_sorted/            # 아동 동화 JSON 데이터셋 (정렬됨)
├── test_data/
│   ├── testset.json        # 휴먼 어노테이션 평가 테스트셋 (61편)
│   └── eval_results.json   # 성능 평가 결과
├── outputs/                # 생성 동화 및 삽화 저장 (자동 생성)
├── frontend/               # 웹 UI (React)
├── main.py                 # CLI 진입점
├── server.py               # FastAPI 서버 (웹 서비스용)
├── evaluate_testset.py     # 성능 평가 스크립트
├── image_generator.py      # Pollinations API 삽화 생성
├── .env                    # API 키 설정
├── .env.example            # 환경변수 예시
└── requirements.txt
```

---

## 파이프라인 흐름

<img width="820" height="700" alt="Image" src="https://github.com/user-attachments/assets/e0e51a90-b53b-45f1-820f-dafc8d7eeefd" />

```
사용자 입력
  ↓
[편향 키워드 검사] ← 왕자·공주·아들·딸 등 성별/인종 편향 단어 거부
  ↓
[Solar Pro 3] → 동화 기획 (교훈 방식·등장인물·결말 방향 설계)
  ↓  ← 퓨샷 참고 동화 (data_sorted 1~20번 고정)
[kanana-1.5-8b] → 동화 본문 생성 (기획 힌트 + 수정 지시 반영)
  ↓
[1차 탐지] kanana-safeguard-8b → 문장 단위 S1~S7 분류
  ├─ HARD (S3·S5·S6): Solar에게 수정 강제 지시
  └─ SOFT (S1·S2·S4·S7): Solar가 맥락 보고 자율 판단
  ↓
[2차 판정] Solar Pro 3 → CSM 6개 항목 품질 평가 + 맥락 안전성 판정
  ├─ PASS → 삽화 생성 (Pollinations API, 기승전결 4장면) → 최종 출력
  └─ FAIL → 수정 지시(rewrite_instructions) 생성 → kanana 재생성 (최대 3회)
```

---

## 설치

```bash
# 1. 가상환경 활성화
conda activate fairytale_llm

# 2. 의존성 설치
pip install -r requirements.txt

# 3. .env 파일 설정
cp .env.example .env
# .env 파일에 아래 키 입력
# UPSTAGE_API_KEY=your_key_here
# STABILITY_API_KEY=your_key_here  (삽화 생성용, 선택)
```

---

## 실행

### CLI 모드

```bash
# 대화형 모드 (권장)
python main.py

# 단일 요청
python main.py --request "편식하는 아이에게 교훈을 주는 동화를 써줘."

# 생성 모델 선택 (기본: 1.5-8b)
python main.py --generator 1.5-8b

# 재작성 모드 선택
python main.py --mode kanana_rewrite   # 기본: Solar 지시 → 카나나 재생성
python main.py --mode solar_rewrite    # Solar가 직접 수정

# 빠른 테스트 (세이프가드 없이)
python main.py --no-safeguard --request "친구와 사이좋게 지내는 동화를 써줘."
```

### 웹 서버

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
# 브라우저에서 http://<서버IP>:8000 접속
```

### 성능 평가

```bash
# 전체 평가 (세이프가드 + Solar + 2단계 비교)
python evaluate_testset.py

# 구조 확인만 (API 호출 없이)
python evaluate_testset.py --dry-run

# 세이프가드 생략 (Solar만)
python evaluate_testset.py --skip-safeguard
```

---

## 평가 기준 (Common Sense Media 기반)

| 항목 | 설명 | 배점 |
|------|------|------|
| 서사적 맥락 | 갈등→현실적 결과 체험→반성→화해 흐름, 인물 일관성 | 1~5점 |
| 아동 모델링 | 따라하고 싶은 긍정적 행동 모델 제시 | 1~5점 |
| 도덕 메시지 | 설교 없이 인과관계로 자연스럽게 교훈 전달 | 1~5점 |
| 편견·고정관념 | 성역할·인종 편견 없음, 중성적 인물 설정 | 1~5점 |
| 언어 표현 | 6~7세 수준 어휘, 대사-인물 일치 | 1~5점 |
| 교육적 가치 | 6~7세가 일상에서 실천 가능한 구체적 행동 모델 | 1~5점 |

**합격 기준: 평균 4.5점 이상 AND 항목별 최저 4.0점 이상**

---

## 세이프가드 카테고리 분기

| 구분 | 카테고리 | 처리 방식 |
|------|----------|-----------|
| **HARD** | S3 성적 콘텐츠, S5 아동 성착취(그루밍), S6 자살·자해 | Solar가 반드시 수정하도록 강제 지시 |
| **SOFT** | S1 증오, S2 괴롭힘, S4 범죄·폭력, S7 잘못된 정보 | Solar가 동화 전체 맥락을 고려하여 자율 판단 |

> S5 카테고리는 그루밍 패턴 탐지 강화를 위해 **LoRA 파인튜닝** 적용

---

## 성능 평가 결과 (테스트셋 61편, 휴먼 어노테이션)

| 지표 | kanana-safeguard-8b 단독 | Solar Pro 3 단독 | **우리 모델 (2단계)** |
|------|--------------------------|------------------|-----------------------|
| Accuracy | 0.6230 | 0.7213 | **0.7869** |
| F1 | 0.6230 | 0.2609 | **0.6667** |
| Precision | 0.4524 | **0.7500** | 0.6500 |
| Recall | 1.0000 | 0.1579 | 0.6842 |
| FPR | 0.5476 | **0.024** | 0.1667 |
| **FNR** ★ | 0.000* | 0.8421 | **0.3158 (균형 최적)** |

> \* kanana-safeguard-8b FNR=0.000은 모든 동화를 UNSAFE로 과탐지한 결과로, 실질적 탐지 능력이 아님.
>
> ★ FNR(False Negative Rate)은 아동 안전 도메인에서 가장 중요한 지표. 유해 콘텐츠를 놓치는 비용이 정상 콘텐츠를 차단하는 비용보다 훨씬 크기 때문.

---

## 퓨샷 데이터 구성

- **퓨샷용 (고정)**: `data_sorted` 폴더 파일명 정렬 기준 **1~20번째** 동화 사용
- **평가용**: 21~81번째 동화 (61편) — 팀원이 직접 SAFE/UNSAFE 태깅
- 태깅 기준: 사람이 읽었을 때 불안감을 줄 수 있는 문장이 하나라도 포함되면 동화 전체 **UNSAFE**

---

## 편향 방지 키워드

다음 단어가 포함된 요청은 거부됩니다:

`왕자` `공주` `아들` `딸` `남자아이` `여자아이` `오빠` `언니` `형` `누나`
`흑인` `백인` `동양인` `서양인` `한국인` `미국인` `중국인` `일본인`

---

## VRAM 사용량

| 모델 | 파라미터 | 예상 VRAM (fp16) |
|------|----------|-----------------|
| kanana-1.5-8b | 8B | ~16GB |
| kanana-safeguard-8b (+ LoRA) | 8B | ~17GB |
| **합계** | ~16B | **~33GB** |

A100 80GB 환경에서 여유 있게 동시 운용 가능.
이미지 생성(SDXL) 시에는 Generator/Safeguard를 CPU로 오프로드 후 SDXL 로드, 생성 완료 후 재로딩하는 방식으로 메모리 관리.

---

## 팀

| 역할 | 이름 | 담당 |
|------|------|------|
| 팀장 | 박소정 | 파이프라인 설계, 프롬프트 설계, 데이터 전처리 |
| 팀원 | 장서연 | 세이프가드 파인튜닝 (S5 그루밍), 이미지 생성 API |
| 팀원 | 박예원 | 프론트엔드 개발, 이미지 생성 API, 발표 자료 |

지도교수: 이웅기 교수 (고려대학교 세종 컴퓨터정보학과)  
산업체 파트너: Upstage · Solar API 조언: 전영훈 PM
