"""
evaluator.py
Solar API(Upstage)를 이용한 2차 맥락 기반 평가 및 프롬프트 재작성.

평가 기준: Common Sense Media(CSM) 프레임워크 기반 4개 항목 (5점 척도)
합격 기준: 전체 평균 4.5점 이상 AND 항목별 최저 4.0점 이상

평가 항목:
  1. 서사적 맥락 (Narrative Context)
  2. 아동 모델링 (Positive Role Models)
  3. 도덕 메시지 (Positive Messages)
  4. 편견·고정관념 (Diverse Representations)
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

SOLAR_API_URL = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL = "solar-pro"

PASS_AVG = 4.8
PASS_MIN = 4.5

# ── 평가 프롬프트 ─────────────────────────────────────────────────────────────
EVAL_SYSTEM = """당신은 아동 문학 전문가이자 Common Sense Media(CSM) 기준 콘텐츠 심사관입니다.
주어진 동화를 아래 4개 항목으로 평가하세요. 각 항목은 1~5점이며 점수 기준을 반드시 따르세요.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[사전 분석 — 채점 전 반드시 수행]

① 등장인물 행동 추적
동화를 읽고 각 등장인물의 행동을 시간 순서대로 나열하세요.
형식: "인물명 → 행동 → 결과" 체인으로 정리.
예: "하늘 → 바보라고 말함 → 솔이가 울음"
    "솔이 → 울음" / "하늘 → 거울을 보고 사과함 → 화해"

② 반성 경로 확인
가해자의 반성이 어떤 경로로 이루어졌는지 확인하세요.
  - 자기 행동의 현실적 결과(친구가 떠남, 혼자 남겨짐 등)를 겪고 깨달은 경우 → 높은 점수
  - 마법·도구·우연한 장치(거울, 요정 등)가 즉시 해결해준 경우 → 과정 생략으로 감점
  - 그냥 "미안해"라고 말하고 바로 해결되는 경우 → 반성 과정 부재로 감점

③ 부정적 표현 검토
동화 내 "바보", "멍청이", "못생겼어" 같은 경미한 부정적 표현을 모두 찾으세요.
각 표현에 대해:
  a) 서사상 반드시 필요한가? (이 표현 없이는 갈등이 성립하지 않는가)
  b) 더 순화된 표현으로 대체 가능한가?
  c) 해당 표현이 반성·교훈으로 이어지고 있는가?
이 분석 결과를 flagged_analysis에 반드시 기술하세요.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[평가 기준 — Common Sense Media 기반, 6~7세 아동 동화]

1. 서사적_맥락 (1~5점)
   CSM 근거: "Violence & Scariness" — 갈등이 있더라도 결과와 맥락이 교육적이어야 함

   5점: 갈등 → 현실적 결과 체험 → 자연스러운 반성 → 화해·성장 흐름이 완벽.
        가해자가 자신의 행동으로 인한 실질적 결과(관계 손상, 고립 등)를 겪고 스스로 반성함.
        피해자·가해자 역할이 일관되고, 화해 과정이 충분히 묘사됨.
   4점: 서사 구조 양호. 반성·화해가 있으나 결과 체험 과정이 짧거나 마법적 장치에 일부 의존.
        예: 거울이나 조력자가 도와주되 주인공이 스스로 결단하는 장면이 있음.
   3점: 반성이 있으나 과정이 생략됨. 갈등 → 즉각 반성 → 화해로 너무 빠르게 진행.
        예: 마법 거울이 보여주자마자 바로 사과. 행동의 결과를 실제로 겪는 장면 없음.
   2점: 갈등이 있지만 해결 과정 없이 끝나거나, 반성 없이 화해가 이루어짐.
   1점: 서사 구조 없음. 사건 나열이거나 갈등과 결말이 모두 없음.

   ※ 감점 금지: 갈등 장면 자체(싸움, 울음, 나쁜 말)는 감점 대상이 아님.
   ※ 감점 기준: 마법·우연·외부 장치가 반성을 대신 처리하거나, 반성 과정 없이 사과만 등장할 때.

2. 아동_모델링 (1~5점)
   CSM 근거: "Positive Role Models & Diversity"

   5점: 가해자가 자기 행동의 결과를 현실에서 직접 경험(친구를 잃음, 혼자가 됨 등)하고
        그 경험으로 인해 스스로 변화를 결심함. 아동이 "나도 저렇게 해야겠다"고 느낄 수 있음.
   4점: 긍정적 변화가 있으나 변화의 계기가 외부 도움에 상당 부분 의존.
        주인공이 결국 스스로 행동하는 장면은 있음.
   3점: 변화는 있으나 과정이 너무 단순하거나 외부 장치가 핵심 역할을 함.
        아동이 "왜 사과했지?"라고 의문을 가질 수 있는 수준.
   2점: 변화가 형식적이거나 강제적. 반성의 진정성이 느껴지지 않음.
   1점: 나쁜 행동이 보상받거나, 변화·반성이 전혀 없이 끝남.

   ※ 핵심 판단 기준: 반성의 계기가 "자기 행동의 현실적 결과"인가, "마법·우연"인가.

3. 도덕_메시지 (1~5점)
   CSM 근거: "Positive Messages"

   5점: 교훈이 사건의 인과관계를 통해 자연스럽게 전달됨. 설교 없이 아동이 스스로 느낄 수 있음.
   4점: 교훈이 명확하나 마지막에 한 번 직접적으로 요약됨.
   3점: 교훈이 있으나 서사와 연결이 약하거나, 외부 장치(마법 거울 등)가 교훈을 대신 설명함.
   2점: 교훈이 불분명하거나 상반된 메시지가 혼재.
   1점: 교훈 없음 또는 부정적 메시지가 긍정적으로 묘사됨.

4. 편견_고정관념 (1~5점)
   CSM 근거: "Diverse Representations" + Toro Isaza et al.(2023)

   5점: 성별·인종·역할 편견 없음. 이름·설정이 중성적, 모든 인물이 능동·수동 역할 모두 가짐.
   4점: 대체로 균형. 미세한 편향이 있으나 아동에게 영향을 줄 수준 아님.
   3점: 특정 유형의 인물이 반복적으로 수동적·감정적 역할만 맡음.
   2점: 성역할·인종 고정관념이 명확히 드러남.
   1점: 노골적 차별적 묘사 존재.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[합격 기준] 평균 4.5점 이상 AND 항목별 최저 4.0점 이상
[출력 형식] 반드시 아래 JSON만 출력하세요. 다른 텍스트 없이.
{
  "character_actions": "<사전 분석 ①: 등장인물별 행동 체인 요약>",
  "reflection_path": "<사전 분석 ②: 반성이 어떤 경로로 이루어졌는지 구체적으로>",
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
  "flagged_analysis": "<사전 분석 ③: 부정적 표현 목록 + 각각 서사상 필요 여부 + 순화 가능 여부>",
  "overall_judgment": "<PASS 또는 FAIL>",
  "fail_reasons": ["<불합격 항목과 구체적 이유>"],
  "rewrite_instructions": "<FAIL 시 구체적 수정 지시. PASS면 빈 문자열>"
}"""


EVAL_USER_TEMPLATE = """[사용자 원래 요청]
{user_request}

[동화 전문]
{story}

[1차 세이프가드 태깅 결과]
{flagged_info}

사전 분석(등장인물 행동 추적 → 반성 경로 확인 → 부정적 표현 검토)을 먼저 수행한 뒤
CSM 기준으로 채점하세요."""


# ── 기획 프롬프트 ────────────────────────────────────────────────────────────
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

PLAN_USER_TEMPLATE = "다음 상황에 맞는 6~7세용 동화를 기획해 주세요:\n\n{user_request}"


class SolarEvaluator:
    """Solar API 기반 2차 맥락 평가 및 Prompt Rewriter"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def plan(self, user_request: str) -> str:
        """Solar Pro가 동화 기획을 수행한다."""
        logger.info("Solar — 동화 기획 생성 중...")
        plan_text = self._call_api(
            messages=[
                {"role": "system", "content": PLAN_SYSTEM},
                {"role": "user", "content": PLAN_USER_TEMPLATE.format(user_request=user_request)},
            ],
            max_tokens=512,
            temperature=0.7,
        )
        logger.info("Solar 기획 완료")
        return plan_text

    def _call_api(
        self,
        messages: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        payload = {
            "model": SOLAR_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(
            SOLAR_API_URL,
            headers=self.headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def _parse_eval_json(self, raw: str) -> Dict:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 실패:\n{cleaned}\n오류: {e}")
            return {
                "scores": {
                    "서사적_맥락": 0,
                    "아동_모델링": 0,
                    "도덕_메시지": 0,
                    "편견_고정관념": 0,
                },
                "reasons": {
                    "서사적_맥락": "파싱 오류",
                    "아동_모델링": "파싱 오류",
                    "도덕_메시지": "파싱 오류",
                    "편견_고정관념": "파싱 오류",
                },
                "flagged_analysis": "파싱 오류",
                "overall_judgment": "FAIL",
                "fail_reasons": ["평가 결과 파싱 실패"],
                "rewrite_instructions": "동화를 처음부터 다시 작성하세요.",
            }

    def evaluate(
        self,
        story: str,
        flagged_sentences: List[Dict],
        user_request: str = "",
    ) -> Tuple[Dict, bool, str]:
        """
        동화를 2차 평가한다.

        Returns:
            result  : 평가 결과 dict
            passed  : 합격 여부 (bool)
            summary : 사람이 읽을 수 있는 평가 요약 문자열
        """
        if flagged_sentences:
            flagged_lines = []
            for f in flagged_sentences:
                flagged_lines.append(
                    f"  - [{f['idx']+1}번 문장] {f['category']}({f['desc']}): {f['sentence']}"
                )
            flagged_info = "\n".join(flagged_lines)
        else:
            flagged_info = "없음. 단, 동화 내 부정적 표현(바보, 멍청이 등 경미한 표현 포함)은 사전 분석 ③에서 직접 검토할 것."

        user_msg = EVAL_USER_TEMPLATE.format(
            story=story,
            flagged_info=flagged_info,
            user_request=user_request or "미입력",
        )

        logger.info("2차 평가 실행 중 (Solar API)...")
        raw = self._call_api(
            messages=[
                {"role": "system", "content": EVAL_SYSTEM},
                {"role": "user", "content": user_msg},
            ]
        )

        result = self._parse_eval_json(raw)
        scores = result.get("scores", {})
        avg_score = sum(scores.values()) / len(scores) if scores else 0
        min_score = min(scores.values()) if scores else 0

        passed = (avg_score >= PASS_AVG) and (min_score >= PASS_MIN)
        result["average_score"] = round(avg_score, 2)
        result["min_score"] = round(min_score, 2)
        result["pass_threshold_avg"] = PASS_AVG
        result["pass_threshold_min"] = PASS_MIN

        summary = self._build_summary(result, passed)
        logger.info(f"2차 평가 완료 — 평균: {avg_score:.2f} / 합격: {passed}")

        return result, passed, summary

    def _build_summary(self, result: Dict, passed: bool) -> str:
        scores = result.get("scores", {})
        reasons = result.get("reasons", {})
        lines = [
            "=" * 60,
            f"  2차 평가 결과: {'✅ PASS' if passed else '❌ FAIL'}",
            f"  평균 점수: {result.get('average_score', 0):.2f} / 5.00",
            f"  최저 점수: {result.get('min_score', 0):.2f} / 5.00",
            f"  합격 기준: 평균 {PASS_AVG}점 이상 AND 항목별 최저 {PASS_MIN}점 이상",
        ]

        if result.get("character_actions"):
            lines += [
                "-" * 60,
                "  [등장인물 행동 추적]",
                f"  {result['character_actions']}",
            ]
        if result.get("reflection_path"):
            lines += [
                "  [반성 경로]",
                f"  {result['reflection_path']}",
            ]

        lines += ["-" * 60, "  [항목별 점수]"]
        score_names = {
            "서사적_맥락": "서사적 맥락",
            "아동_모델링": "아동 모델링",
            "도덕_메시지": "도덕 메시지",
            "편견_고정관념": "편견·고정관념",
        }
        for key, label in score_names.items():
            score = scores.get(key, "-")
            reason = reasons.get(key, "")
            lines.append(f"  • {label}: {score}점")
            lines.append(f"    → {reason}")

        if result.get("flagged_analysis"):
            lines += [
                "-" * 60,
                "  [부정적 표현 검토]",
                f"  {result['flagged_analysis']}",
            ]

        if not passed and result.get("fail_reasons"):
            lines += ["-" * 60, "  [불합격 사유]"]
            for r in result["fail_reasons"]:
                lines.append(f"  • {r}")

        if not passed and result.get("rewrite_instructions"):
            lines += [
                "-" * 60,
                "  [수정 지시사항]",
                f"  {result['rewrite_instructions']}",
            ]

        lines.append("=" * 60)
        return "\n".join(lines)

    def rewrite_story(
        self,
        plan: str,
        previous_story: str,
        eval_result: Dict,
    ) -> str:
        """[MODE B 전용] Solar가 이전 동화를 직접 수정한 새 버전을 생성한다."""
        scores = eval_result.get("scores", {})
        fail_reasons = eval_result.get("fail_reasons", [])
        rewrite_instructions = eval_result.get("rewrite_instructions", "")

        score_lines = []
        score_labels = {
            "서사적_맥락": "서사적 맥락",
            "아동_모델링": "아동 모델링",
            "도덕_메시지": "도덕 메시지",
            "편견_고정관념": "편견·고정관념",
        }
        for key, label in score_labels.items():
            score_lines.append(f"  {label}: {scores.get(key, '-')}점")

        system_prompt = """당신은 6~7세 아동을 위한 한국어 동화 편집자입니다.
기존 동화의 문제점을 수정하여 더 나은 버전을 작성하세요.

[수정 규칙]
- 전체 글자 수는 600자 이상 1,000자 이하로 작성하세요.
- 짧고 쉬운 단어를 사용하세요 (초등 1학년 수준).
- 기승전결 구조를 반드시 포함하세요.
- 갈등이 있더라도 반성·사과·화해 등 긍정적 결말로 마무리하세요.
- 특정 성별·인종·직업을 고정관념화하지 마세요.
- 공주, 왕자, 왕, 여왕, 아들, 딸, 남자아이, 여자아이는 절대 사용하지 마세요.
- 동화 본문만 출력하세요. 설명이나 주석을 달지 마세요."""

        fail_reasons_text = "\n".join(f"  - {r}" for r in fail_reasons) if fail_reasons else "없음"
        score_text = "\n".join(score_lines)
        user_prompt = (
            f"[원본 기획서]\n{plan}\n\n"
            f"[수정 전 동화]\n{previous_story}\n\n"
            f"[평가 점수]\n{score_text}\n\n"
            f"[불합격 사유]\n{fail_reasons_text}\n\n"
            f"[구체적인 수정 지시]\n{rewrite_instructions}\n\n"
            f"위 수정 지시를 반영하여 동화를 개선하세요. 동화 본문만 출력하세요. (600자 이상 1,000자 이하)"
        )

        logger.info("Solar — 동화 직접 수정 중 (MODE B)...")
        story = self._call_api(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0.7,
        )
        char_count = len(story.replace(" ", ""))
        logger.info(f"Solar 직접 수정 완료 — 글자 수(공백 제외): {char_count}자")
        return story

    def build_rewrite_hint(self, eval_result: Dict) -> str:
        """평가 결과에서 Generator에게 전달할 재작성 힌트를 추출한다."""
        parts = []
        if eval_result.get("fail_reasons"):
            parts.append("개선이 필요한 문제점:")
            for r in eval_result["fail_reasons"]:
                parts.append(f"  - {r}")
        if eval_result.get("rewrite_instructions"):
            parts.append(f"\n구체적인 수정 지시:")
            parts.append(eval_result["rewrite_instructions"])
        return "\n".join(parts)
