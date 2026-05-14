"""
evaluator.py
Solar API로 동화를 맥락 기반 평가하고, 기준 미달 시 직접 첨삭합니다.

평가 기준 (CSM 기반 5개 항목, 각 1~5점 전 구간 정의):
  1. positive_message    긍정적 교훈이 서사에 자연스럽게 녹아있는가
  2. role_model          아이가 따라할 만한 긍정적 행동이 묘사되는가
  3. violence_scariness  갈등/폭력 장면이 교육적 맥락 안에 있는가
  4. diverse_repr        성별·외모·직업 고정관념이 없는가
  5. narrative_integrity 캐릭터 행동이 맥락상 타당하고 기승전결이 있는가

통과 기준:
  - 5개 항목 평균 4.0점 이상
  - 항목별 최저점 3점 이상
  - narrative_integrity 3점 미만이면 즉시 FAIL (하드 룰)
"""

import json
import os
import re
from dataclasses import dataclass
from openai import OpenAI

PASS_AVERAGE = 4.5
PASS_MIN_SCORE = 3
NARRATIVE_HARD_MIN = 3  # 서사 무결성 최저 기준

EVALUATION_SYSTEM_PROMPT = """당신은 6~7세 아동 동화 전문 평가자입니다.
아래 순서대로 분석하고, 반드시 JSON 형식으로만 응답하세요.

[STEP 1 - 캐릭터-행동 맵 작성 (평가 전 필수)]
각 등장인물의 행동을 시간 순서대로 정리하세요.
각 행동에 대해 "왜 그 행동을 했는가?"를 함께 기록하세요.

형식: "캐릭터명": ["행동1 (이유)", "행동2 (이유)", ...]

[STEP 2 - 요청 반영 확인]
사용자가 요청한 나쁜 행동이 동화에 실제로 등장하는지 확인하세요.

[STEP 3 - 갈등 원인 분석]
갈등의 원인이 한쪽에만 있는지, 양쪽 모두에게 있는지 분석하세요.
- 양쪽 모두 잘못이 있는 경우, 각자 자신의 잘못을 인정하고 사과했는지 확인하세요.
- 한쪽만 일방적으로 사과하고 끝났다면 narrative_integrity와 role_model 점수를 낮게 주세요.

[STEP 4 - 5개 항목 평가 (각 1~5점, 전 구간 사용)]
★ 중요: 만점(5점)이 아닌 경우 반드시 구체적인 이유를 score_reasons에 기술하세요.
  - 어떤 장면/문장이 문제인지
  - 무엇이 빠졌는지
  - 어떻게 바꾸면 5점이 될 수 있는지

1. positive_message (긍정적 교훈)
   5점: 교훈이 사건 흐름에서 자연스럽게 도출됨. 설명 없이도 아이가 스스로 느낄 수 있음
   4점: 교훈이 있으나 마지막에 설명적으로 직접 서술됨 (예: "민수는 배웠어요")
   3점: 교훈이 있으나 억지스럽거나 서사와 따로 노는 느낌
   2점: 교훈이 불명확하거나 부정적 암시가 있음
   1점: 교훈 없음. 나쁜 행동이 처벌 없이 끝나거나 오히려 보상받음

2. role_model (긍정적 행동 모델링)
   5점: 사과·나눔·협력 등 아이가 따라하고 싶은 행동이 구체적 장면으로 묘사됨
   4점: 긍정 행동이 있으나 짧거나 추상적으로 처리됨
   3점: 긍정 행동은 있으나 부정 행동과 균형이 맞지 않음
   2점: 긍정 행동이 결과 없이 끝나거나 거의 없음
   1점: 부정 행동만 있고 따라할 만한 행동 모델 없음

3. violence_scariness (갈등의 교육적 맥락)
   5점: 갈등이 교육적 해결로 이어지고, 나쁜 행동의 결과가 구체적으로 묘사됨
   4점: 갈등 후 해결되나 과정이 다소 급하게 처리됨
   3점: 갈등 있고 해결되나 연결 맥락이 약함
   2점: 갈등이 해결되지 않거나 결과가 불분명
   1점: 갈등/폭력이 해결 없이 끝나거나 정당화됨

4. diverse_repr (다양성·편견 없음)
   5점: 성별·외모·직업 관련 편견 없음. 중립적 표현만 사용
   4점: 경미한 고정관념 표현 1개, 전반적으로 무난
   3점: 고정관념 표현 2~3개, 서사에 큰 영향은 없음
   2점: 명확한 편견 표현이 여러 곳에 등장
   1점: 성역할·외모 고정관념이 서사의 핵심 요소로 작동

5. narrative_integrity (서사 무결성)
   캐릭터 행동이 앞뒤 맥락상 타당한지, 기승전결이 완결되는지 평가합니다.
   STEP 1의 행동 맵과 STEP 3의 갈등 원인 분석을 반드시 참고하세요.

   5점: 모든 행동에 명확한 이유가 있음. 기승전결 완결. 갈등 원인에 따라 적절한 책임 인정
   4점: 전반적으로 자연스럽지만 인과관계가 약하거나 책임 인정이 부분적인 곳이 있음
   3점: 구조는 있으나 인과관계 불명확한 곳이 여러 곳이거나, 갈등 원인 제공자가 책임을 지지 않음
   2점: 인과관계 단절이 심함. 한쪽만 일방적으로 사과하며 갈등 원인이 무시됨
   1점: 서사 구조 없음. 또는 요청한 상황 자체가 동화에 없음

   [즉시 FAIL 조건 - 다른 점수와 무관하게 pass: false]
   ① 나쁜 행동을 한 캐릭터가 사과하지 않고 끝남
   ② 가해자가 사과하는 계기(이유)가 전혀 서술되지 않음
   ③ 요청한 나쁜 행동이 동화에 아예 없음
   ④ 갈등 원인이 양쪽에 있는데 한쪽만 사과하고 상대방의 잘못은 완전히 무시됨

[응답 형식 - JSON만 출력, 다른 텍스트 없음]
{
  "character_action_map": {
    "<캐릭터명>": ["행동1 (이유)", "행동2 (이유)"]
  },
  "request_match": {
    "requested_bad_action": "<요청한 나쁜 행동>",
    "is_present": <true/false>,
    "note": "<한 줄 설명>"
  },
  "conflict_analysis": {
    "cause": "<갈등 원인 요약>",
    "both_at_fault": <true/false>,
    "fault_details": "<양쪽 잘못이 있다면 각각 무엇인지. 한쪽만이라면 null>",
    "both_apologized": <true/false | null>
  },
  "scores": {
    "positive_message": <1~5>,
    "role_model": <1~5>,
    "violence_scariness": <1~5>,
    "diverse_repr": <1~5>,
    "narrative_integrity": <1~5>
  },
  "score_reasons": {
    "positive_message": "<점수 근거: 만점이 아니라면 구체적으로 어떤 장면/문장이 문제인지 명시>",
    "role_model": "<점수 근거: 만점이 아니라면 무엇이 부족한지 명시>",
    "violence_scariness": "<점수 근거: 만점이 아니라면 어떤 부분이 급하게 처리됐는지 명시>",
    "diverse_repr": "<점수 근거: 만점이 아니라면 어떤 편견 표현이 있는지 명시>",
    "narrative_integrity": "<점수 근거: 만점이 아니라면 어떤 인과관계가 약한지, 누가 책임을 안 졌는지 명시>"
  },
  "apology_check": {
    "bad_actor": "<나쁜 행동을 한 캐릭터>",
    "apology_trigger": "<사과하게 된 계기. 없으면 none>",
    "who_apologized_first": "<먼저 사과한 캐릭터>",
    "is_reversed": <true/false>
  },
  "average": <소수점 1자리>,
  "pass": <true/false>,
  "overall_feedback": "<전체 동화 한 줄 평가>",
  "rewrite_instruction": "<FAIL 시 구체적 수정 지시. PASS면 null>"
}
"""


@dataclass
class EvaluationResult:
    scores: dict
    average: float
    passed: bool
    overall_feedback: str
    rewrite_instruction: str | None
    raw_response: str
    character_action_map: dict = None
    request_match: dict = None
    score_reasons: dict = None
    apology_check: dict = None
    conflict_analysis: dict = None  # 갈등 원인 분석 (양측 잘못 여부)


class SolarEvaluator:
    def __init__(self):
        api_key = os.getenv("UPSTAGE_API_KEY")
        if not api_key:
            raise ValueError("UPSTAGE_API_KEY 환경변수가 없습니다.")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.upstage.ai/v1/solar",
        )
        self.model = "solar-pro"

    # ── 평가 ──────────────────────────────────────────────────────────────
    def evaluate(
        self,
        fairy_tale_text: str,
        original_request: str,
        calibration_sample: dict = None,
    ) -> EvaluationResult:
        """
        동화를 평가합니다.

        Args:
            calibration_sample: data_loader.get_calibration_sample() 반환값.
                                 제공 시 평가 기준점으로 사용.
        """
        calib_block = ""
        if calibration_sample:
            calib_block = f"""
[평가 기준점 - 실제 아동 동화 예시]
제목: {calibration_sample['title']}
단어 수: {calibration_sample['word_count']}개
내용: {calibration_sample['text'][:400]}...
(위 동화를 5점짜리 기준으로 삼아 상대적으로 평가하세요)
"""

        user_message = f"""[사용자 요청]
{original_request}

[평가할 동화]
{fairy_tale_text}
{calib_block}
위 동화를 평가 기준에 따라 평가해주세요."""

        print("[Evaluator] Solar 평가 중...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=2500,
        )
        raw = response.choices[0].message.content.strip()
        return self._parse_response(raw)

    # ── Solar 직접 첨삭 ────────────────────────────────────────────────────
    def rewrite_tale(
        self,
        original_tale: str,
        evaluation: EvaluationResult,
        original_request: str,
    ) -> str:
        """
        Solar가 동화를 직접 첨삭합니다.
        캐릭터·배경·전체 줄거리는 유지하고 문제 문장만 최소한으로 수정합니다.
        """
        # 문제 위치 특정
        apology = evaluation.apology_check or {}
        apology_info = ""
        if apology:
            apology_info = f"""
[사과 구조 분석]
- 나쁜 행동 캐릭터: {apology.get('bad_actor', '?')}
- 먼저 사과한 캐릭터: {apology.get('who_apologized_first', '?')}
- 역전 구조: {'예' if apology.get('is_reversed') else '아니오'}
- 사과 계기: {apology.get('apology_trigger', 'none')}"""

        system_prompt = """당신은 아동 동화 편집자입니다.
원본 동화의 캐릭터·배경·전체 줄거리는 반드시 유지하고,
지적된 문제 부분의 문장만 최소한으로 수정하세요.

수정 규칙:
1. 나쁜 행동을 한 캐릭터가 직접 사과해야 합니다
2. 사과하게 된 감정 변화 계기를 한 문장으로 명시하세요
   예: "친구가 우는 모습을 보고 마음이 아팠어요. 그래서 사과했어요."
3. 피해자가 먼저 사과하는 장면은 삭제하고 가해자 사과로 교체하세요
4. 서술문은 "~했어요", "~였어요" 높임말로 작성하세요
5. 수정 후 동화 전체를 출력하세요 (설명 없이 동화만)"""

        user_message = f"""[원본 동화]
{original_tale}

[수정 지시]
{evaluation.rewrite_instruction}
{apology_info}

[사용자 원래 요청]
{original_request}

위 지시에 따라 최소한으로 수정한 동화 전체를 출력하세요."""

        print("[Evaluator] Solar 첨삭 중...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=2500,
        )
        return response.choices[0].message.content.strip()

    # ── 카나나 재생성용 프롬프트 (fallback) ───────────────────────────────
    def build_rewrite_prompt(
        self,
        original_request: str,
        evaluation: EvaluationResult,
    ) -> str:
        score_lines = "\n".join(
            f"  - {k}: {v}점 ({evaluation.score_reasons.get(k, '')})"
            for k, v in evaluation.scores.items()
        ) if evaluation.score_reasons else "\n".join(
            f"  - {k}: {v}점" for k, v in evaluation.scores.items()
        )
        return f"""[원래 요청]
{original_request}

[이전 동화의 문제점]
{evaluation.overall_feedback}

[항목별 점수]
{score_lines}

[구체적 수정 지시]
{evaluation.rewrite_instruction}

위 문제를 해결하여 처음부터 새 동화를 써주세요.
나쁜 행동을 한 캐릭터가 직접 사과하고, 왜 사과했는지 반드시 서술하세요.
"""

    # ── 내부 유틸 ─────────────────────────────────────────────────────────
    def _parse_response(self, raw: str) -> EvaluationResult:
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

        # 1차: 완전한 JSON
        try:
            data = json.loads(cleaned)
            return self._build_result(data, raw)
        except json.JSONDecodeError:
            pass

        # 2차: scores만 추출
        scores_match = re.search(r'"scores"\s*:\s*\{([^}]+)\}', cleaned, re.DOTALL)
        if scores_match:
            try:
                scores = json.loads("{" + scores_match.group(1) + "}")
                avg_match = re.search(r'"average"\s*:\s*([\d.]+)', cleaned)
                average = float(avg_match.group(1)) if avg_match else (
                    sum(scores.values()) / len(scores)
                )
                pass_match = re.search(r'"pass"\s*:\s*(true|false)', cleaned, re.I)
                passed = (pass_match.group(1).lower() == "true") if pass_match else (
                    average >= PASS_AVERAGE and min(scores.values(), default=0) >= PASS_MIN_SCORE
                )
                fb_match = re.search(r'"overall_feedback"\s*:\s*"([^"]*)"', cleaned)
                rw_match = re.search(r'"rewrite_instruction"\s*:\s*"([^"]*)"', cleaned)
                print("[Evaluator] 부분 파싱 성공")
                return EvaluationResult(
                    scores=scores, average=round(float(average), 2), passed=passed,
                    overall_feedback=fb_match.group(1) if fb_match else "(응답 잘림)",
                    rewrite_instruction=rw_match.group(1) if rw_match else None,
                    raw_response=raw,
                )
            except Exception:
                pass

        # 3차: 실패
        print(f"[Evaluator] JSON 파싱 실패:\n{raw[:300]}")
        return EvaluationResult(
            scores={k: 0 for k in ["positive_message","role_model",
                                    "violence_scariness","diverse_repr","narrative_integrity"]},
            average=0.0, passed=False,
            overall_feedback="평가 파싱 실패",
            rewrite_instruction="동화를 다시 생성해주세요.",
            raw_response=raw,
        )

    def _build_result(self, data: dict, raw: str) -> EvaluationResult:
        scores = data.get("scores", {})
        average = data.get("average", sum(scores.values()) / len(scores) if scores else 0)

        # narrative_integrity 하드 룰
        narrative_score = scores.get("narrative_integrity", 5)
        apology_check = data.get("apology_check", {})
        is_reversed = apology_check.get("is_reversed", False)

        if narrative_score < NARRATIVE_HARD_MIN or is_reversed:
            passed = False
            if is_reversed:
                print(f"[Evaluator] ❌ 하드룰 FAIL: 사과 역전 "
                      f"(가해자={apology_check.get('bad_actor')}, "
                      f"먼저 사과={apology_check.get('who_apologized_first')})")
            if narrative_score < NARRATIVE_HARD_MIN:
                print(f"[Evaluator] ❌ 하드룰 FAIL: narrative_integrity {narrative_score}점")
        else:
            passed = data.get("pass",
                average >= PASS_AVERAGE and
                min(scores.values(), default=0) >= PASS_MIN_SCORE
            )

        # 요청 미반영 하드 룰
        request_match = data.get("request_match", {})
        if not request_match.get("is_present", True):
            passed = False
            print(f"[Evaluator] ❌ 하드룰 FAIL: 요청 상황 미반영 "
                  f"({request_match.get('note', '')})")

        # 양측 잘못인데 한쪽만 사과한 경우 하드 룰
        conflict = data.get("conflict_analysis", {})
        if (conflict.get("both_at_fault") and
                conflict.get("both_apologized") is False):
            passed = False
            print(f"[Evaluator] ❌ 하드룰 FAIL: 양측 잘못인데 한쪽만 사과 "
                  f"({conflict.get('fault_details', '')})")

        return EvaluationResult(
            scores=scores,
            average=round(float(average), 2),
            passed=passed,
            overall_feedback=data.get("overall_feedback", ""),
            rewrite_instruction=data.get("rewrite_instruction"),
            raw_response=raw,
            character_action_map=data.get("character_action_map"),
            request_match=request_match,
            score_reasons=data.get("score_reasons"),
            apology_check=apology_check,
            conflict_analysis=conflict,
        )