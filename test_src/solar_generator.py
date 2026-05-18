"""
test_src/solar_generator.py
Solar Pro가 동화 기획과 본문 생성을 담당하는 Generator.
FairyTaleGenerator와 동일한 인터페이스를 가져 Pipeline에 바로 대체 가능.
"""

import logging
import re
from typing import Dict, List

import requests

from src.evaluator import PLAN_SYSTEM, PLAN_USER_TEMPLATE
from src.generator import STORY_SYSTEM

logger = logging.getLogger(__name__)

SOLAR_API_URL = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL   = "solar-pro"


class SolarGenerator:
    """Solar Pro 기반 동화 기획·본문 생성기 (FairyTaleGenerator 대체용)."""

    model_id = SOLAR_MODEL

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _call_api(
        self,
        messages: List[Dict],
        max_tokens: int = 2048,
        temperature: float = 0.8,
    ) -> str:
        payload = {
            "model": SOLAR_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(SOLAR_API_URL, headers=self.headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def plan(self, user_request: str) -> str:
        """동화 기획서를 생성한다."""
        logger.info("Solar Pro — 동화 기획 생성 중...")
        plan_text = self._call_api(
            messages=[
                {"role": "system", "content": PLAN_SYSTEM},
                {"role": "user",   "content": PLAN_USER_TEMPLATE.format(user_request=user_request)},
            ],
            max_tokens=512,
            temperature=0.7,
        )
        logger.info("Solar Pro 기획 완료")
        return plan_text

    def generate(self, plan: str, rewrite_hint: str = "", few_shot_examples: str = "") -> str:
        """기획서를 받아 동화 본문을 생성한다."""
        few_shot_section = ""
        if few_shot_examples:
            few_shot_section = f"\n\n[참고 동화 — 아래 동화들의 문체·구조·길이를 참고하세요]\n{few_shot_examples}"

        hint_section = ""
        if rewrite_hint:
            hint_section = f"\n\n[이전 평가 피드백 — 반드시 반영하세요]\n{rewrite_hint}"

        user_prompt = (
            f"아래 기획서를 바탕으로 6~7세용 한국어 동화 본문을 작성하세요.\n\n"
            f"[동화 기획서]\n{plan}"
            f"{few_shot_section}"
            f"{hint_section}\n\n"
            f"지금 바로 동화 본문만 작성하세요. (700자 이상 1,800자 이하)"
        )

        logger.info("Solar Pro — 동화 본문 생성 중...")
        story = self._call_api(
            messages=[
                {"role": "system", "content": STORY_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=2048,
            temperature=0.8,
        )

        # 마지막 완성 문장까지만 반환
        story = _trim_to_last_sentence(story)
        char_count = len(story.replace(" ", ""))
        logger.info(f"Solar Pro 생성 완료 — 글자 수(공백 제외): {char_count}자")
        return story


def _trim_to_last_sentence(text: str) -> str:
    stripped = text.strip()
    if not stripped or stripped[-1] in ".!?":
        return stripped
    m = re.search(r"(.*[.!?])", stripped, re.DOTALL)
    if m:
        trimmed = m.group(1).strip()
        logger.warning(f"문장 끝 잘림 감지 → 트리밍 적용 ({len(stripped)}자 → {len(trimmed)}자)")
        return trimmed
    return stripped
