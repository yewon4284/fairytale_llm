"""
main_solar_only.py
Solar Pro가 기획 → 생성 → 평가 → 직접 수정까지 전부 담당하는 파이프라인.

비교 실험 목적:
  main.py (MODE A) : Solar 기획 + 카나나 생성 + Solar 평가·수정 지시 + 카나나 재생성
  main_solar_only.py: Solar 기획 + Solar 생성 + Solar 평가 + Solar 직접 수정

주의:
  생성자와 평가자가 동일 모델(Solar)이므로 평가 독립성은 낮음.
  단, "Solar 단독 vs 카나나+Solar 협업" 동화 품질 비교 실험에 활용.

실행:
  python main_solar_only.py
  python main_solar_only.py --request "친구를 때리면 안된다는 교훈을 주는 동화를 써줘."
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple

import requests
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
load_dotenv()

# ── 상수 ──────────────────────────────────────────────────────────────────────
SOLAR_API_URL   = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL     = "solar-pro"
MAX_ATTEMPTS    = 4
PASS_AVG        = 4.5
PASS_MIN        = 4.0

# ── 프롬프트 ──────────────────────────────────────────────────────────────────
PLAN_SYSTEM = """당신은 아동 교육 전문가이자 동화 작가입니다.
사용자가 설명한 상황을 바탕으로, 6~7세 아동에게 교훈을 전달하는 동화를 기획하세요.

[전달 방식 옵션]
- 역지사지: 주인공이 피해자 입장을 직접 경험하며 깨닫는 방식
- 제3자 조언: 현명한 조력자(동물, 나무, 친구 등)가 주인공에게 가르침을 주는 방식
- 결과 체험: 잘못된 행동의 결과를 주인공이 직접 겪으며 반성하는 방식
- 감정 공감: 상대방의 감정을 느끼고 이해하는 과정을 통해 변화하는 방식

[제약 사항]
- 주인공 이름은 성별이 드러나지 않는 이름으로 설정 (예: 도담, 하늘, 솔이, 누리, 봄이)
- 공주/왕자/왕/여왕/아들/딸/남자아이/여자아이 사용 금지
- 특정 인종·직업·지역 고정관념 없이 설정

[출력 형식] 반드시 아래 형식 그대로 출력하세요:
[핵심 교훈] <한 문장으로 명확하게>
[전달 방식] <위 4가지 중 선택한 방식 + 이 방식을 선택한 이유 한 문장>
[주인공 설정] <이름(성별 중립) + 처한 상황 한 문장>
[조력자/계기] <변화를 이끄는 존재 또는 사건 한 문장>
[결말 방향] <어떤 깨달음이나 변화로 마무리되는지 한 문장>
[동화 분위기] <따뜻한 / 유쾌한 / 진지한 / 모험적인 중 선택>"""

STORY_SYSTEM = """당신은 6~7세 아동을 위한 한국어 동화 작가입니다.
주어진 기획서를 바탕으로 동화 본문만 작성하세요.

[작성 규칙]
- 전체 글자 수는 600자 이상 1,000자 이하로 작성하세요 (공백 제외).
- 짧고 쉬운 단어를 사용하세요 (초등 1학년 수준).
- 기승전결 구조를 반드시 포함하세요.
- 갈등 이후 반성·사과·화해 등 긍정적 결말로 마무리하세요.
- 특정 성별·인종·직업을 고정관념화하지 마세요.
- 폭력·혐오·성적 표현을 사용하지 마세요.
- 주인공 이름은 성별이 드러나지 않는 이름을 사용하세요.
- 공주, 왕자, 왕, 여왕, 아들, 딸, 남자아이, 여자아이는 사용하지 마세요.
- 동화 본문만 출력하세요. 설명이나 제목을 달지 마세요."""

EVAL_SYSTEM = """당신은 아동 문학 전문가이자 Common Sense Media(CSM) 기준 콘텐츠 심사관입니다.
주어진 동화를 아래 4개 항목으로 평가하세요.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[사전 분석 — 채점 전 반드시 수행]
① 등장인물 행동 추적: 각 인물의 행동을 "인물 → 행동 → 결과" 체인으로 정리
② 반성 경로 확인: 반성이 자기 행동의 현실적 결과인지, 마법·우연인지 명시
③ 부정적 표현 검토: 동화 내 부정적 표현의 서사적 필요성·순화 가능성 검토
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[평가 기준]

1. 서사적_맥락 (1~5점)
   5점: 갈등 → 현실적 결과 체험 → 자연스러운 반성 → 화해·성장 흐름 완벽
   4점: 서사 구조 양호. 반성·화해 있으나 결과 체험 과정이 짧거나 장치에 일부 의존
   3점: 반성 있으나 과정 생략. 갈등 → 즉각 반성 → 화해로 너무 빠름
   2점: 갈등이 있지만 해결 과정 없이 끝남
   1점: 서사 구조 없음
   ※ 마법·우연·외부 장치가 반성을 대신 처리하면 감점

2. 아동_모델링 (1~5점)
   5점: 가해자가 행동의 현실적 결과를 직접 경험하고 스스로 변화를 결심
   4점: 긍정적 변화 있으나 외부 도움에 상당 부분 의존
   3점: 변화는 있으나 외부 장치가 핵심 역할
   2점: 변화가 형식적이거나 강제적
   1점: 변화·반성 없이 끝남

3. 도덕_메시지 (1~5점)
   5점: 교훈이 사건의 인과관계를 통해 자연스럽게 전달. 설교 없음
   4점: 교훈 명확하나 마지막에 한 번 직접 요약됨
   3점: 교훈 있으나 서사와 연결 약함
   2점: 교훈 불분명하거나 상반된 메시지 혼재
   1점: 교훈 없음 또는 부정적 메시지

4. 편견_고정관념 (1~5점)
   5점: 성별·인종·역할 편견 없음. 이름·설정 중성적
   4점: 대체로 균형. 미세한 편향이나 아동에게 영향 줄 수준 아님
   3점: 특정 유형 인물이 반복적으로 수동적 역할만 맡음
   2점: 고정관념 명확히 드러남
   1점: 노골적 차별 묘사

[합격 기준] 평균 4.5점 이상 AND 항목별 최저 4.0점 이상
[출력 형식] 반드시 아래 JSON만 출력하세요. 다른 텍스트 없이.
{
  "character_actions": "<등장인물 행동 체인 요약>",
  "reflection_path": "<반성 경로 구체적으로>",
  "scores": {
    "서사적_맥락": <1~5 정수>,
    "아동_모델링": <1~5 정수>,
    "도덕_메시지": <1~5 정수>,
    "편견_고정관념": <1~5 정수>
  },
  "reasons": {
    "서사적_맥락": "<몇 점 기준 해당 + 이유 + 왜 더 높지 않은지>",
    "아동_모델링": "<몇 점 기준 해당 + 이유 + 왜 더 높지 않은지>",
    "도덕_메시지": "<몇 점 기준 해당 + 이유 + 왜 더 높지 않은지>",
    "편견_고정관념": "<몇 점 기준 해당 + 이유 + 왜 더 높지 않은지>"
  },
  "flagged_analysis": "<부정적 표현 목록 + 서사적 필요 여부 + 순화 가능 여부>",
  "overall_judgment": "<PASS 또는 FAIL>",
  "fail_reasons": ["<불합격 항목과 구체적 이유>"],
  "rewrite_instructions": "<FAIL 시 구체적 수정 지시. PASS면 빈 문자열>"
}"""

REWRITE_SYSTEM = """당신은 6~7세 아동을 위한 한국어 동화 편집 전문가입니다.
이전 동화의 문제점을 반영하여 개선된 동화를 직접 작성하세요.

[수정 규칙]
- 전체 글자 수는 600자 이상 1,000자 이하로 작성하세요 (공백 제외).
- 짧고 쉬운 단어를 사용하세요 (초등 1학년 수준).
- 기승전결 구조를 반드시 포함하세요.
- 갈등 이후 반성·사과·화해 등 긍정적 결말로 마무리하세요.
- 특정 성별·인종·직업을 고정관념화하지 마세요.
- 공주, 왕자, 왕, 여왕, 아들, 딸, 남자아이, 여자아이는 사용하지 마세요.
- 동화 본문만 출력하세요. 설명이나 제목을 달지 마세요."""


# ── Solar API 호출 ────────────────────────────────────────────────────────────
def call_solar(api_key: str, messages: List[Dict],
               max_tokens: int = 1024, temperature: float = 0.5) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": SOLAR_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(SOLAR_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ── 단계별 함수 ───────────────────────────────────────────────────────────────
def solar_plan(api_key: str, user_request: str) -> str:
    """Solar가 동화 기획을 수립한다."""
    logger.info("Solar — 동화 기획 중...")
    return call_solar(
        api_key,
        messages=[
            {"role": "system", "content": PLAN_SYSTEM},
            {"role": "user",   "content": f"다음 상황에 맞는 6~7세용 동화를 기획해 주세요:\n\n{user_request}"},
        ],
        max_tokens=512,
        temperature=0.7,
    )


def solar_generate(api_key: str, plan: str) -> str:
    """Solar가 기획서를 바탕으로 동화 본문을 생성한다."""
    logger.info("Solar — 동화 본문 생성 중...")
    story = call_solar(
        api_key,
        messages=[
            {"role": "system", "content": STORY_SYSTEM},
            {"role": "user",   "content": f"아래 기획서를 바탕으로 동화 본문을 작성하세요.\n\n[기획서]\n{plan}\n\n동화 본문만 작성하세요. (600자 이상 1,000자 이하, 공백 제외)"},
        ],
        max_tokens=1024,
        temperature=0.8,
    )
    char_count = len(story.replace(" ", ""))
    logger.info(f"생성 완료 — 글자 수(공백 제외): {char_count}자")
    return story


def solar_evaluate(api_key: str, story: str,
                   user_request: str, flagged_info: str) -> Dict:
    """Solar가 동화를 CSM 기준으로 평가한다."""
    import json, re
    logger.info("Solar — 2차 평가 중...")

    user_msg = (
        f"[사용자 원래 요청]\n{user_request}\n\n"
        f"[동화 전문]\n{story}\n\n"
        f"[1차 세이프가드 태깅 결과]\n{flagged_info}\n\n"
        "사전 분석(행동 추적 → 반성 경로 → 부정 표현 검토)을 먼저 수행한 뒤 CSM 기준으로 채점하세요."
    )

    raw = call_solar(
        api_key,
        messages=[
            {"role": "system", "content": EVAL_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=1500,
        temperature=0.2,
    )

    # JSON 파싱 — 큰따옴표 이스케이프 문제 방어
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # 문자열 내 이스케이프 안 된 큰따옴표를 작은따옴표로 교체 후 재시도
        fixed = re.sub(r'(?<!\\)"(?=[^:,\{\}\[\]\n])', "'", cleaned)
        try:
            result = json.loads(fixed)
        except json.JSONDecodeError:
            logger.error(f"JSON 파싱 실패:\n{cleaned}")
            return {
                "scores": {"서사적_맥락": 0, "아동_모델링": 0, "도덕_메시지": 0, "편견_고정관념": 0},
                "reasons": {"서사적_맥락": "파싱 오류", "아동_모델링": "파싱 오류",
                            "도덕_메시지": "파싱 오류", "편견_고정관념": "파싱 오류"},
                "character_actions": "파싱 오류",
                "reflection_path": "파싱 오류",
                "flagged_analysis": "파싱 오류",
                "overall_judgment": "FAIL",
                "fail_reasons": ["평가 결과 파싱 실패"],
                "rewrite_instructions": "동화의 반성 과정이 부족합니다. 주인공이 행동의 현실적 결과를 직접 겪도록 수정하고 600자 이상으로 작성하세요.",
            }

    scores = result.get("scores", {})
    avg = sum(scores.values()) / len(scores) if scores else 0
    result["average_score"] = round(avg, 2)
    result["passed"] = (avg >= PASS_AVG) and (min(scores.values(), default=0) >= PASS_MIN)
    return result


def solar_rewrite(api_key: str, plan: str, previous_story: str, eval_result: Dict) -> str:
    """Solar가 평가 결과를 반영해 동화를 직접 수정한다."""
    scores   = eval_result.get("scores", {})
    reasons  = eval_result.get("reasons", {})
    fails    = eval_result.get("fail_reasons", [])
    instruct = eval_result.get("rewrite_instructions", "")
    char_count = len(previous_story.replace(" ", ""))

    score_summary = "\n".join(
        f"  {k}: {v}점 — {reasons.get(k, '')}" for k, v in scores.items()
    )
    fail_summary = "\n".join(f"  - {r}" for r in fails) if fails else "없음"

    user_msg = (
        f"[원본 기획서]\n{plan}\n\n"
        f"[수정 전 동화 ({char_count}자)]\n{previous_story}\n\n"
        f"[항목별 점수]\n{score_summary}\n\n"
        f"[불합격 사유]\n{fail_summary}\n\n"
        f"[구체적 수정 지시]\n{instruct}\n\n"
        f"위 내용을 반영하여 동화를 개선하세요. 동화 본문만 출력하세요. (600자 이상 1,000자 이하, 공백 제외)"
    )

    logger.info("Solar — 동화 직접 수정 중...")
    story = call_solar(
        api_key,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=1024,
        temperature=0.7,
    )
    char_count = len(story.replace(" ", ""))
    logger.info(f"수정 완료 — 글자 수(공백 제외): {char_count}자")
    return story


def print_eval_summary(eval_result: Dict, attempt: int):
    scores  = eval_result.get("scores", {})
    reasons = eval_result.get("reasons", {})
    passed  = eval_result.get("passed", False)
    avg     = eval_result.get("average_score", 0)

    print(f"\n{'='*60}")
    print(f"  2차 평가 결과 (시도 {attempt}): {'✅ PASS' if passed else '❌ FAIL'}")
    print(f"  평균 점수: {avg:.2f} / 5.00  (기준: 평균 {PASS_AVG} AND 항목별 최저 {PASS_MIN})")

    if eval_result.get("character_actions"):
        print(f"{'─'*60}")
        print(f"  [행동 추적] {eval_result['character_actions']}")
    if eval_result.get("reflection_path"):
        print(f"  [반성 경로] {eval_result['reflection_path']}")

    print(f"{'─'*60}")
    print("  [항목별 점수]")
    labels = {"서사적_맥락": "서사적 맥락", "아동_모델링": "아동 모델링",
              "도덕_메시지": "도덕 메시지", "편견_고정관념": "편견·고정관념"}
    for key, label in labels.items():
        print(f"  • {label}: {scores.get(key, '-')}점")
        print(f"    → {reasons.get(key, '')}")

    if eval_result.get("flagged_analysis"):
        print(f"{'─'*60}")
        print(f"  [부정적 표현 검토] {eval_result['flagged_analysis']}")

    if not passed:
        if eval_result.get("fail_reasons"):
            print(f"{'─'*60}")
            print("  [불합격 사유]")
            for r in eval_result["fail_reasons"]:
                print(f"  • {r}")
        if eval_result.get("rewrite_instructions"):
            print(f"{'─'*60}")
            print(f"  [수정 지시] {eval_result['rewrite_instructions']}")

    print(f"{'='*60}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def get_user_request(args) -> str:
    if args.request:
        return args.request.strip()

    print("\n" + "=" * 70)
    print("  🌟 Solar Only — 동화 생성 시스템 (비교 실험용)")
    print("=" * 70)
    print("\n어떤 교훈을 담은 동화를 원하시나요?")
    print("예시: '친구에게 욕설을 하면 안된다는 교훈을 주는 동화를 써줘.'")
    print()
    print("⚠  편향 단어 주의: 공주/왕자/아들/딸/특정 성별·인종 단어는 입력하지 마세요.")
    print("-" * 70)
    while True:
        req = input("요청 입력: ").strip()
        if req:
            return req
        print("요청을 입력해 주세요.")


def main():
    parser = argparse.ArgumentParser(description="Solar Only 동화 생성 (비교 실험용)")
    parser.add_argument("--request", type=str, default=None)
    args = parser.parse_args()

    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        logger.error("UPSTAGE_API_KEY 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    user_request = get_user_request(args)

    # 편향 단어 검사 (generator.py 의존 없이 독립 실행 가능하도록 인라인)
    import re
    BIASED = {"아들","딸","공주","왕자","왕","여왕","남자아이","여자아이","남자","여자",
              "오빠","언니","형","누나","흑인","백인","동양인","서양인","한국인","미국인","중국인","일본인"}
    bias_pat = re.compile(
        r"(?<![가-힣a-zA-Z])(" + "|".join(re.escape(w) for w in sorted(BIASED, key=len, reverse=True)) + r")(?![가-힣a-zA-Z])",
        re.UNICODE,
    )
    m = bias_pat.search(user_request)
    if m:
        print(f"\n❌ 편향 단어 감지: '{m.group()}' — 성별·인종 단어 없이 상황만 설명해 주세요.")
        sys.exit(0)

    # 세이프가드는 카나나 로컬 모델이라 이 파일에서는 생략
    # (동화 생성·평가 품질 비교가 목적이므로 세이프가드 없이 진행)
    print("\n⚠  이 실험 파일은 카나나 세이프가드 없이 Solar만으로 실행됩니다.")
    print("   (세이프가드 포함 정식 실행은 main.py 를 사용하세요.)\n")

    print("=" * 70)
    print("  🌟 Solar Only 동화 생성 시스템 (비교 실험용)")
    print("=" * 70)
    print(f"  모델:    Solar Pro (기획·생성·평가·수정 전담)")
    print(f"  최대 시도: {MAX_ATTEMPTS}회")
    print(f"\n📝 사용자 요청:\n  {user_request}")
    print("=" * 70)

    # Step 1: 기획 (1회차만)
    print("\n🧩 [Step 1] Solar — 동화 기획")
    plan = solar_plan(api_key, user_request)
    print(plan)

    records = []
    final_story = ""
    passed = False

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n{'─'*70}")
        print(f"  🎯 시도 {attempt} / {MAX_ATTEMPTS}")
        print(f"{'─'*70}")

        # Step 2: 생성 or 수정
        if attempt == 1:
            print("\n📝 [Step 2] Solar — 동화 본문 생성")
            story = solar_generate(api_key, plan)
        else:
            print(f"\n✏️  [Step 2] Solar — 동화 직접 수정 (시도 {attempt})")
            story = solar_rewrite(api_key, plan, records[-1]["story"], records[-1]["eval"])

        final_story = story
        print("\n📖 [생성된 동화]")
        print(story)
        print(f"\n  글자 수 (공백 제외): {len(story.replace(' ', ''))}자")

        # Step 3: 평가
        print("\n🧠 [Step 3] Solar — 평가")
        flagged_info = "세이프가드 미실행 (Solar Only 실험 모드)"
        eval_result = solar_evaluate(api_key, story, user_request, flagged_info)
        print_eval_summary(eval_result, attempt)

        passed = eval_result.get("passed", False)
        records.append({"attempt": attempt, "story": story, "eval": eval_result})

        if passed:
            print(f"\n✅ {attempt}회 시도에서 합격!")
            break

        if attempt < MAX_ATTEMPTS:
            print(f"\n🔄 수정 지시 반영 → {attempt+1}회차 시작...")
        else:
            print(f"\n⚠ 최대 시도 횟수({MAX_ATTEMPTS}회) 초과. 마지막 동화를 출력합니다.")

    # 최종 출력
    print("\n" + "=" * 70)
    print("  🏁 최종 결과")
    print("=" * 70)
    print(f"  상태:  {'✅ 합격' if passed else '❌ 불합격 (최대 시도 초과)'}")
    print(f"  모델:  Solar Pro (단독)")
    print(f"  총 시도: {len(records)}회")
    print("\n🧩 [동화 기획]")
    print(plan)
    print("\n📖 [최종 동화]")
    print("─" * 70)
    print(final_story)
    print("─" * 70)
    print(f"  글자 수 (공백 제외): {len(final_story.replace(' ', ''))}자")
    print("\n📊 [시도별 점수 히스토리]")
    for rec in records:
        avg = rec["eval"].get("average_score", "-")
        status = "✅ PASS" if rec["eval"].get("passed") else "❌ FAIL"
        print(f"  시도 {rec['attempt']}: 평균 {avg}점 → {status}")
    print("=" * 70)


if __name__ == "__main__":
    main()