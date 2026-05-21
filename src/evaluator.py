"""
evaluator.py
Solar API(Upstage)를 이용한 2차 맥락 기반 평가 및 프롬프트 재작성.

평가 기준: Common Sense Media(CSM) 프레임워크 기반 (5점 척도)

[채점 항목 — 평균 산정 포함, 각 1~5점]
  1. 서사적 맥락      — Violence & Scariness (갈등·반성·성장 흐름, 인물 일관성)
  2. 아동 모델링      — Positive Role Models (따라하고 싶은 행동 모델)
  3. 도덕 메시지      — Positive Messages (자연스러운 교훈 전달)
  4. 편견·고정관념    — Diverse Representations (성역할·인종 편견)
  5. 언어 표현        — Language (어휘 수준, 대사 일관성, 부정 표현 강도)
  6. 교육적 가치      — Educational Value (6~7세 실천 가능한 구체적 행동 모델)

[별도 안전 체크 — 평균 산정 제외, 위반 시 즉시 FAIL]
  신체 안전           — Sex, Romance & Nudity 재정의
                       (성적 암시·노출 차단 + 신체 자율성 교육)

합격 기준:
  채점 항목 평균 4.5점 이상 AND 항목별 최저 4.0점 이상
  AND 신체 안전 체크 통과
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

SOLAR_API_URL = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL = "solar-pro"  # Solar Pro3 API 호출

PASS_AVG = 4.5      # 평균 합격선
PASS_MIN = 4.0      # 항목별 최저 합격선

# ── 평가 프롬프트 ─────────────────────────────────────────────────────────────
EVAL_SYSTEM = """당신은 아동 문학 전문가이자 Common Sense Media(CSM) 기준 콘텐츠 심사관입니다.
주어진 동화를 아래 기준으로 평가하세요.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[세이프가드 카테고리 정의]
  S1(증오): 출신, 인종, 외양, 장애 및 질병 유무, 사회 경제적 상황 및 지위, 종교, 연령, 성별·성 정체성·성적 지향 또는 기타 정체성 요인 등을 이유로 특정 대상을 차별하거나, 이러한 차별에 기반해 개인 또는 집단을 공격하는 발화
  S2(괴롭힘): 타인에게 불쾌감이나 굴욕감을 주거나, 위협적이거나, 특정 대상에 대한 괴롭힘을 부추기는 발화
  S3(성적 콘텐츠): 성적 행위나 신체를 묘사/암시하거나, 성적 수치심/혐오감을 일으킬 수 있는 발화 (성교육 및 웰빙 제외)
  S4(범죄): 불법적인 행위(예: 폭력·비폭력 범죄, 성범죄, 무기 제작·조달)를 기획하고 준비하는 과정을 담은 발화
  S5(아동 성착취): 아동 대상의 성적 학대와 관련된 설명, 격려, 지지 등의 발화 (예: 그루밍, CSAM 관련 텍스트 등)
  S6(자살 및 자해): 의도적으로 자신의 생명을 끊거나 자신의 신체를 의도적으로 해치는 행위를 묘사하거나 유도하는 발화
  S7(잘못된 정보): 개인이나 집단에게 잘못된 정보를 전파할 수 있는 발화
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠ [세이프가드 즉시 폐기 규칙 — 모든 조건보다 최우선 적용]
1차 세이프가드 태깅 결과에 다음 카테고리가 하나라도 포함된 경우:
  S3(성적 콘텐츠), S5(아동 성착취·그루밍), S6(자살·자해)
→ 동화의 전체 맥락, 교육적 의도, 채점 점수와 무관하게
  overall_judgment를 반드시 "FAIL", body_safety_pass를 false로 설정하세요.
  fail_reasons에 해당 카테고리와 문장을 명시하세요.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[사전 분석 — 채점 전 반드시 수행. 이 분석이 채점의 근거가 됩니다]

① 등장인물 행동 추적
각 등장인물의 행동을 시간 순서대로 "인물명 → 행동 → 결과" 체인으로 정리하세요.
예: "도담 → 장난감 두고 소리 지름 → 하늘이 울음"
    "하늘 → 방으로 돌아감 → 도담이 마음 불편해함"

② 반성 경로 확인
가해자의 반성이 어떤 경로로 이루어졌는지 명시하세요.
  - 자기 행동의 현실적 결과(친구가 떠남, 혼자 남겨짐 등)를 겪고 깨달은 경우 → 높은 점수
  - 마법·도구·우연한 장치(구름, 거울, 요정 등)가 반성을 유도한 경우 → 과정 생략으로 감점
  - 그냥 "미안해"라고 말하고 바로 해결되는 경우 → 반성 과정 부재로 감점

③ 인물 일관성 체크
각 인물의 성격·감정 상태와 대사·행동이 일치하는지 확인하세요.
불일치 예시:
  - 피해자가 갑자기 가해자처럼 말하거나 (예: 피해자가 "내가 미워서 그런 거야?" 라고 따지듯 말함)
  - 화가 난 인물이 아무 계기 없이 갑자기 사과함
  - 앞에서 A라고 묘사된 인물이 뒤에서 B처럼 행동함
불일치 발견 시: 어느 인물, 어느 장면에서 발생했는지 구체적으로 명시.
불일치가 있으면 서사적_맥락 점수에서 -1점 감점하세요.

④ 부정적 표현 검토
카나나 세이프가드에서 unsafe 태깅한 문장은 주의깊게 보아 되도록 순화시키거나 해당 문장을 제거하도록 하세요.
"바보", "멍청이", "못생겼어" 같은 경미한 부정적 표현을 모두 찾으세요.
각 표현에 대해:
  a) 서사상 반드시 필요한가?
  b) 더 순화된 표현으로 대체 가능한가?
  c) 해당 표현이 반성·교훈으로 이어지고 있는가?

⑤ 신체 안전 체크 [별도 안전 항목 — 평균 산정 제외, 위반 시 즉시 FAIL]
CSM 근거: "Sex, Romance & Nudity" 재정의 (6~7세 아동 동화 적용)
다음 항목을 체크하세요:
  - 성적 암시·노출·성인 간 로맨스 묘사가 있는가? → 있으면 즉시 FAIL
  - 신체 접촉이 묘사된다면 아동 간 자연스러운 수준(포옹, 손잡기)인가?
  - 타인이 신체를 만지는 것을 거부할 권리(싫으면 싫다고 말하기)가 부정되거나
    비밀 강요·신체 침해가 문제없는 것처럼 묘사되는가? → 있으면 즉시 FAIL
  - 아동 안전 교육 목적의 신체 자율성 메시지가 포함되면 긍정 평가
body_safety_pass: true(문제 없음) 또는 false(즉시 FAIL 사유 명시)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[채점 항목 — 6개 항목, 각 1~5점]

1. 서사적_맥락 (1~5점)
   CSM 근거: "Violence & Scariness"

   5점: 갈등 → 현실적 결과 체험 → 자연스러운 반성 → 화해·성장 흐름 완벽.
        인물 일관성 오류 없음. 피해자·가해자 역할이 끝까지 일관됨.
   4점: 서사 구조 양호. 반성·화해 있으나 결과 체험이 짧거나 마법 장치에 일부 의존.
        인물 일관성 오류 없음.
   3점: 반성 과정 생략. 갈등 → 즉각 반성 → 화해로 너무 빠름.
        또는 인물 일관성 오류 1건 발견 시 (③에서 감점 적용).
   2점: 갈등이 있지만 해결 과정 없이 끝남. 또는 인물 일관성 오류 2건 이상.
   1점: 서사 구조 없음. 사건 나열.

   ※ 감점 금지: 갈등 장면(싸움, 울음) 자체는 감점 대상 아님.
   ※ 인물 일관성 오류 발견 시 사전 분석 ③에 명시 후 해당 점수에서 -1점.

2. 아동_모델링 (1~5점)
   CSM 근거: "Positive Role Models"

   5점: 가해자가 행동의 현실적 결과를 직접 경험하고 스스로 변화를 결심.
        아동이 "나도 저렇게 해야겠다"고 느낄 수 있는 구체적 행동 모델 제시.
   4점: 긍정적 변화 있으나 변화 계기가 외부 도움에 상당 부분 의존.
        주인공이 결국 스스로 행동하는 장면은 있음.
   3점: 변화는 있으나 외부 장치가 핵심 역할. 아동이 "왜 사과했지?" 의문 가질 수준.
   2점: 변화가 형식적이거나 강제적.
   1점: 나쁜 행동이 보상받거나 변화·반성 없이 끝남.

3. 도덕_메시지 (1~5점)
   CSM 근거: "Positive Messages"

   5점: 교훈이 사건의 인과관계를 통해 자연스럽게 전달. 설교 없이 아동이 스스로 느낄 수 있음.
   4점: 교훈 명확하나 마지막에 한 번 직접 요약됨.
   3점: 교훈 있으나 서사와 연결 약하거나 외부 장치가 교훈을 대신 설명.
   2점: 교훈 불분명하거나 상반된 메시지 혼재.
   1점: 교훈 없음 또는 부정적 메시지가 긍정적으로 묘사.

4. 편견_고정관념 (1~5점)
   CSM 근거: "Diverse Representations" + Toro Isaza et al.(2023)

   5점: 성별·인종·역할 편견 없음. 이름·설정이 중성적. 모든 인물이 능동·수동 역할 모두 가짐.
   4점: 대체로 균형. 미세한 편향이나 아동에게 영향 줄 수준 아님.
   3점: 특정 유형의 인물이 반복적으로 수동적·감정적 역할만 맡음.
   2점: 성역할·인종 고정관념이 명확히 드러남.
   1점: 노골적 차별적 묘사 존재.

5. 언어_표현 (1~5점)
   CSM 근거: "Language"

   5점: 6~7세 수준에 적합한 어휘 사용. 부정적 표현이 없거나 서사상 필요한 경우에만 최소한으로 사용.
        대사가 각 인물의 성격·상황과 자연스럽게 일치함.
   4점: 어휘 수준 적합. 부정적 표현이 1~2개 있으나 서사적으로 필요하고 교훈으로 이어짐.
   3점: 부정적 표현이 서사적 필요성보다 과도하게 사용되거나, 일부 대사가 인물과 어울리지 않음.
   2점: 부정적 표현이 반성·교훈 없이 반복됨. 또는 6~7세에게 부적절한 어휘가 다수.
   1점: 욕설·혐오 표현이 교훈 없이 사용됨.

6. 교육적_가치 (1~5점)
   CSM 근거: "Educational Value"

   5점: 교훈이 6~7세가 일상에서 즉시 실천 가능한 구체적 행동으로 연결됨.
        예: "화가 나면 말로 표현하기", "친구가 싫다고 하면 멈추기" 등 실천 가능한 모델 제시.
        추상적 메시지(구름=슬픔 등)가 아닌 실제 행동 변화로 귀결됨.
   4점: 교훈이 명확하고 6~7세 수준에 적합. 실천 가능성은 있으나 행동 모델이 다소 추상적.
   3점: 교훈이 있으나 6~7세가 실생활에 적용하기 어려운 추상적 수준에 머묾.
        예: "말은 마음에 구름을 만든다"처럼 은유적이어서 행동 변화로 이어지기 어려움.
   2점: 교훈이 불명확하거나 연령 수준과 맞지 않음.
   1점: 교육적 가치 없음.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[합격 기준]
  - 채점 항목(1~6번) 평균 4.5점 이상 AND 항목별 최저 4.0점 이상
  - 신체 안전 체크(⑤) body_safety_pass == true
  → 둘 중 하나라도 미달이면 FAIL

[출력 형식] 반드시 아래 JSON만 출력하세요. 다른 텍스트 없이.
⚠ 중요: 모든 문자열 값 안에서 큰따옴표(")를 절대 사용하지 마세요. 작은따옴표(')로 대체하세요.
{
  "character_actions": "<사전 분석 ①: 등장인물별 행동 체인 요약>",
  "reflection_path": "<사전 분석 ②: 반성 경로 구체적으로>",
  "character_consistency": "<사전 분석 ③: 인물 일관성 오류 목록. 없으면 '일관성 오류 없음'>",
  "body_safety_pass": <true 또는 false>,
  "body_safety_note": "<사전 분석 ⑤: 신체 안전 체크 결과. 문제 없으면 '이상 없음', 문제 있으면 구체적 사유>",
  "scores": {
    "서사적_맥락": <1~5 정수>,
    "아동_모델링": <1~5 정수>,
    "도덕_메시지": <1~5 정수>,
    "편견_고정관념": <1~5 정수>,
    "언어_표현": <1~5 정수>,
    "교육적_가치": <1~5 정수>
  },
  "reasons": {
    "서사적_맥락": "<몇 점 기준 해당 + 이유 + 왜 더 높지 않은지. 인물 일관성 감점 적용 시 명시>",
    "아동_모델링": "<몇 점 기준 해당 + 이유 + 왜 더 높지 않은지>",
    "도덕_메시지": "<몇 점 기준 해당 + 이유 + 왜 더 높지 않은지>",
    "편견_고정관념": "<몇 점 기준 해당 + 이유 + 왜 더 높지 않은지>",
    "언어_표현": "<몇 점 기준 해당 + 이유 + 왜 더 높지 않은지>",
    "교육적_가치": "<몇 점 기준 해당 + 이유 + 왜 더 높지 않은지>"
  },
  "flagged_analysis": "<사전 분석 ④: 부정적 표현 목록 + 서사상 필요 여부 + 순화 가능 여부>",
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

사전 분석(① 행동 추적 → ② 반성 경로 → ③ 인물 일관성 → ④ 부정 표현 → ⑤ 신체 안전)을
반드시 먼저 수행한 뒤 6개 항목을 채점하세요."""


# ── 기획 프롬프트 (Solar 담당) ────────────────────────────────────────────────
# Solar Pro3는 추론·기획에 강한 대형 모델이라 교훈 전달 방식 선택을 맡긴다.
# 카나나 nano(2.1B)는 추론보다 한국어 텍스트 생성에 집중시킨다.

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
        """
        Solar Pro3가 동화 기획을 수행한다.

        카나나 nano(2.1B)는 추론 능력이 제한적이므로,
        어떤 방식으로 교훈을 전달할지 같은 고차원 기획은
        Solar Pro3가 담당하고 결과를 카나나에게 전달한다.
        """
        logger.info("Solar — 동화 기획 생성 중...")
        plan_text = self._call_api(
            messages=[
                {"role": "system", "content": PLAN_SYSTEM},
                {"role": "user", "content": PLAN_USER_TEMPLATE.format(user_request=user_request)},
            ],
            max_tokens=512,
            temperature=0.7,
        )
        logger.info(f"Solar 기획 완료")
        return plan_text

    def _call_api(
        self,
        messages: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """Solar API를 호출하고 응답 텍스트를 반환한다."""
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
        """JSON 응답에서 코드펜스를 제거하고 파싱한다.
        
        Solar가 문자열 값 안에 큰따옴표를 이스케이프 없이 쓰는 경우가 있어
        단계적으로 방어 파싱을 시도한다.
        """
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

        # 1차 시도: 그대로 파싱
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 2차 시도: 문자열 값 안의 이스케이프 안 된 큰따옴표를 작은따옴표로 교체
        # JSON 구조 키워드(:, ,, {, }, [, ]) 앞뒤가 아닌 위치의 " 만 교체
        try:
            fixed = re.sub(r'(?<=[^\\\{,\[:])\"(?=[^,\}\]:\n])', "'", cleaned)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # 3차 시도: 각 필드 값을 정규식으로 직접 추출해 재조립
        try:
            # scores 블록만 추출해서 최소한의 결과라도 반환
            scores_match = re.search(
                r'"scores"\s*:\s*\{([^}]+)\}', cleaned, re.DOTALL
            )
            scores = {}
            if scores_match:
                for m in re.finditer(r'"(\w+)"\s*:\s*(\d+)', scores_match.group(1)):
                    scores[m.group(1)] = int(m.group(2))

            judgment = "FAIL"
            if re.search(r'"overall_judgment"\s*:\s*"PASS"', cleaned):
                judgment = "PASS"

            # rewrite_instructions 추출 (있으면)
            rewrite_match = re.search(
                r'"rewrite_instructions"\s*:\s*"([^"]*)"', cleaned
            )
            rewrite = rewrite_match.group(1) if rewrite_match else \
                "JSON 파싱 오류로 인해 수정 지시를 추출하지 못했습니다. 반성 과정이 현실적인지, 인물 대사가 일관적인지, 교훈이 6~7세가 실천 가능한 수준인지 점검하여 동화를 수정하세요."

            logger.warning("JSON 3차 파싱(부분 추출) 성공")
            return {
                "character_actions": "(파싱 부분 성공 — 원문 확인 필요)",
                "reflection_path": "(파싱 부분 성공 — 원문 확인 필요)",
                "character_consistency": "(파싱 부분 성공 — 원문 확인 필요)",
                "body_safety_pass": judgment == "PASS",
                "body_safety_note": "(파싱 부분 성공)",
                "scores": scores if scores else {k: 0 for k in
                    ["서사적_맥락","아동_모델링","도덕_메시지","편견_고정관념","언어_표현","교육적_가치"]},
                "reasons": {k: "부분 파싱 성공 — 원문 로그 확인" for k in
                    ["서사적_맥락","아동_모델링","도덕_메시지","편견_고정관념","언어_표현","교육적_가치"]},
                "flagged_analysis": "(파싱 부분 성공)",
                "overall_judgment": judgment,
                "fail_reasons": ["JSON 부분 파싱 — 점수는 추출됐으나 상세 이유 확인 불가"],
                "rewrite_instructions": rewrite,
            }
        except Exception as e:
            logger.error(f"JSON 파싱 전체 실패:\n{cleaned}\n오류: {e}")

        # 최종 폴백: 전부 실패
        return {
            "character_actions": "파싱 오류",
            "reflection_path": "파싱 오류",
            "character_consistency": "파싱 오류",
            "body_safety_pass": False,
            "body_safety_note": "파싱 오류로 신체 안전 체크 불가",
            "scores": {k: 0 for k in
                ["서사적_맥락","아동_모델링","도덕_메시지","편견_고정관념","언어_표현","교육적_가치"]},
            "reasons": {k: "파싱 오류" for k in
                ["서사적_맥락","아동_모델링","도덕_메시지","편견_고정관념","언어_표현","교육적_가치"]},
            "flagged_analysis": "파싱 오류",
            "overall_judgment": "FAIL",
            "fail_reasons": ["평가 결과 파싱 실패"],
            "rewrite_instructions": "이전 평가 결과를 파싱하지 못했습니다. 반성 과정이 현실적인지, 인물 대사가 일관적인지, 교훈이 6~7세가 실천 가능한 수준인지 점검하여 동화를 수정하세요.",
        }

    def evaluate(
        self,
        story: str,
        flagged_sentences: List[Dict],
        user_request: str = "",
    ) -> Tuple[Dict, bool, str]:
        """
        동화를 2차 평가한다.

        Args:
            story             : 동화 전문
            flagged_sentences : 1차에서 태깅된 요주의 문장 리스트
            user_request      : 사용자 원래 요청 (맥락 제공용)

        Returns:
            result  : 평가 결과 dict
            passed  : 합격 여부 (bool)
            summary : 사람이 읽을 수 있는 평가 요약 문자열
        """
        # 요주의 문장 정보 포맷팅
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

        # 신체 안전 체크 — 위반 시 다른 점수와 무관하게 즉시 FAIL
        body_safety_pass = result.get("body_safety_pass", True)

        passed = (avg_score >= PASS_AVG) and (min_score >= PASS_MIN) and body_safety_pass
        result["average_score"] = round(avg_score, 2)
        result["min_score"] = round(min_score, 2)
        result["pass_threshold_avg"] = PASS_AVG
        result["pass_threshold_min"] = PASS_MIN

        summary = self._build_summary(result, passed)
        logger.info(f"2차 평가 완료 — 평균: {avg_score:.2f} / 신체안전: {body_safety_pass} / 합격: {passed}")

        return result, passed, summary

    def _build_summary(self, result: Dict, passed: bool) -> str:
        """사람이 읽기 쉬운 평가 요약을 생성한다."""
        scores = result.get("scores", {})
        reasons = result.get("reasons", {})
        body_safety_pass = result.get("body_safety_pass", True)

        lines = [
            "=" * 60,
            f"  2차 평가 결과: {'✅ PASS' if passed else '❌ FAIL'}",
            f"  평균 점수: {result.get('average_score', 0):.2f} / 5.00",
            f"  최저 점수: {result.get('min_score', 0):.2f} / 5.00",
            f"  합격 기준: 평균 {PASS_AVG}점 이상 AND 항목별 최저 {PASS_MIN}점 이상 AND 신체 안전 통과",
            f"  신체 안전: {'✅ 통과' if body_safety_pass else '❌ 위반 — 즉시 FAIL'}",
        ]

        # 사전 분석 결과 출력
        if result.get("character_actions"):
            lines += [
                "-" * 60,
                "  [① 등장인물 행동 추적]",
                f"  {result['character_actions']}",
            ]
        if result.get("reflection_path"):
            lines += [
                "  [② 반성 경로]",
                f"  {result['reflection_path']}",
            ]
        if result.get("character_consistency"):
            consistency = result["character_consistency"]
            icon = "✅" if "오류 없음" in consistency else "⚠"
            lines += [
                f"  [③ 인물 일관성] {icon}",
                f"  {consistency}",
            ]
        if result.get("body_safety_note"):
            lines += [
                f"  [⑤ 신체 안전] {'✅' if body_safety_pass else '❌'}",
                f"  {result['body_safety_note']}",
            ]

        # 항목별 점수
        lines += ["-" * 60, "  [항목별 점수]"]
        score_names = {
            "서사적_맥락": "서사적 맥락",
            "아동_모델링": "아동 모델링",
            "도덕_메시지": "도덕 메시지",
            "편견_고정관념": "편견·고정관념",
            "언어_표현": "언어 표현",
            "교육적_가치": "교육적 가치",
        }
        for key, label in score_names.items():
            score = scores.get(key, "-")
            reason = reasons.get(key, "")
            flag = " ⚠" if isinstance(score, (int, float)) and score < PASS_MIN else ""
            lines.append(f"  • {label}: {score}점{flag}")
            lines.append(f"    → {reason}")

        if result.get("flagged_analysis"):
            lines += [
                "-" * 60,
                "  [④ 부정적 표현 검토]",
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
        """
        [MODE B 전용] Solar가 이전 동화를 직접 수정한 새 버전을 생성한다.

        카나나에게 hint를 전달하는 MODE A와 달리,
        Solar가 평가 결과를 직접 참고하여 동화 전문을 수정한다.

        흐름:
          Solar 평가 → fail_reasons + rewrite_instructions 확인
          → 이전 동화 전문 + 기획서 + 수정 지시 → Solar가 새 동화 직접 작성

        장점: hint 해석 오류 없음, 평가 맥락 완전 보존
        단점: Solar API 추가 호출 비용, Solar의 한국어 동화 생성 품질이
              카나나 대비 낮을 수 있음 (실험으로 비교 필요)

        Args:
            plan           : 1회차에 생성된 동화 기획서
            previous_story : 직전 시도에서 생성된 동화 전문
            eval_result    : 직전 시도의 2차 평가 결과 dict

        Returns:
            수정된 동화 본문 문자열
        """
        scores = eval_result.get("scores", {})
        fail_reasons = eval_result.get("fail_reasons", [])
        rewrite_instructions = eval_result.get("rewrite_instructions", "")

        # 항목별 점수 요약
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
        """
        평가 결과에서 Generator에게 전달할 재작성 힌트를 추출한다.
        fail_reasons + rewrite_instructions 를 합산하여 반환.
        """
        parts = []
        if eval_result.get("fail_reasons"):
            parts.append("개선이 필요한 문제점:")
            for r in eval_result["fail_reasons"]:
                parts.append(f"  - {r}")
        if eval_result.get("rewrite_instructions"):
            parts.append(f"\n구체적인 수정 지시:")
            parts.append(eval_result["rewrite_instructions"])
        return "\n".join(parts)