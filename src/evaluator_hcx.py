"""
evaluator_hcx.py
Naver CLOVA Studio(HyperCLOVA X, HCX-007)로 2차 평가를 돌리기 위한 어댑터.

SolarEvaluator를 상속해서 프롬프트(EVAL_SYSTEM/EVAL_USER_TEMPLATE), JSON 파싱
(_parse_eval_json), 채점/합격 판정 로직(evaluate())을 그대로 재사용하고,
_call_api()만 CLOVA Studio API 호출 방식으로 오버라이드한다. 이렇게 해야
Solar Pro3 / Solar Pro4 / HCX-007을 같은 프롬프트·같은 합격기준으로 순수하게
모델만 바꿔서 비교할 수 있다 (안 그러면 "무엇 때문에 차이가 났는지" 알 수 없음).

2026-08-18 실제 API 호출로 검증 완료 (pod에서 curl + python 양쪽 확인):
  - 응답 스키마는 후보 1번 {"result": {"message": {"content": "..."}}} 이 맞음.
  - HCX-007은 추론(inference) 모델이라 v3 Chat Completions의 일반 `maxTokens`
    파라미터를 거부한다 ("Invalid parameter: maxTokens", maxTokens=50과 1024
    둘 다 실패). 반드시 `maxCompletionTokens`를 대신 써야 함 — 공식 문서
    (clovastudio-chatcompletionsv3-fc)에 "maxCompletionTokens: 추론 모델용,
    maxTokens와 동시 사용 불가"로 명시돼있고, 실제로 maxCompletionTokens=1024로
    호출하니 정상 응답(status 20000) 받음.

사용법 (evaluator.py의 SolarEvaluator 대신):
    from src.evaluator_hcx import NaverHCXEvaluator
    evaluator = NaverHCXEvaluator(api_key=HCX_API_KEY, model="HCX-007")
"""

import json
import logging

import requests

from src.evaluator import SolarEvaluator

logger = logging.getLogger(__name__)

# v3 Chat Completions 엔드포인트 (모델명을 경로에 포함)
HCX_API_URL_TEMPLATE = "https://clovastudio.stream.ntruss.com/v3/chat-completions/{model}"


class NaverHCXEvaluator(SolarEvaluator):
    """HyperCLOVA X(HCX-007) 기반 2차 평가. SolarEvaluator와 인터페이스 동일."""

    def __init__(self, api_key: str, model: str = "HCX-007", eval_temperature: float = 0.0):
        # 부모 __init__은 Solar용 헤더를 만드므로 호출하지 않고 직접 설정.
        self.api_key = api_key
        self.model = model
        self.eval_temperature = eval_temperature
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _call_api(self, messages, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        url = HCX_API_URL_TEMPLATE.format(model=self.model)
        payload = {
            "messages": messages,
            "temperature": temperature,
            "topP": 0.8,
            # HCX-007은 추론 모델이라 maxTokens가 아니라 maxCompletionTokens를 써야
            # 함(2026-08-18 실제 호출로 확인 — maxTokens는 값에 상관없이 거부됨).
            "maxCompletionTokens": max_tokens,
            # 구조화 출력(우리 프롬프트가 요구하는 JSON)과 thinking 모드는 동시 사용이
            # 안 된다고 확인됨(2026-08-17 조사) — 우리 프롬프트는 이미 자체적으로
            # 단계별 사전분석을 텍스트로 시키고 있어 모델 내장 thinking이 굳이
            # 필요 없으므로 꺼둠.
            "thinking": {"effort": "none"},
        }
        resp = requests.post(url, headers=self.headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # 응답 스키마 검증 완료(2026-08-18): {"result": {"message": {"content": "..."}}} 가 맞음.
        try:
            return data["result"]["message"]["content"].strip()
        except (KeyError, TypeError):
            pass
        # 혹시 모를 OpenAI 호환 포맷 폴백 (v3 OpenAI-compatibility 엔드포인트를 쓰게
        # 되는 경우를 대비한 안전장치, 평소엔 안 탐).
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError):
            pass

        logger.error(f"HCX 응답 형식을 못 알아봄 — 원본: {json.dumps(data, ensure_ascii=False)[:500]}")
        raise ValueError(
            "HCX API 응답 파싱 실패 — 응답 JSON 구조가 예상과 다름. "
            "_call_api()의 후보 1/2 분기를 실제 응답 구조에 맞게 수정할 것."
        )
