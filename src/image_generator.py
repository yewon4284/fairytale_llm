"""
image_generator.py
동화 최종본에서 핵심 장면을 추출하고 Pollinations.ai로 이미지를 생성한다.

[흐름]
  1. Solar Pro — 동화(한국어)에서 핵심 장면 4개 추출 + 영어 프롬프트 생성
  2. Pollinations.ai — 장면별 이미지 생성 (무료, nologo=false)
  3. outputs/ 디렉터리에 PNG 저장 + Pollinations URL 반환
"""

import json
import logging
import os
import re
import urllib.parse
from typing import Dict, List, Tuple

import requests

logger = logging.getLogger(__name__)

SOLAR_API_URL      = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL        = "solar-pro"
POLLINATIONS_URL   = "https://image.pollinations.ai/prompt/{prompt}"
DEFAULT_OUTPUT_DIR = "outputs"


# ── 장면 추출 프롬프트 ────────────────────────────────────────────────────────
SCENE_SYSTEM = """You are a children's book art director.
Extract exactly 4 key scenes from a Korean fairy tale and write a Pollinations.ai image prompt for each.

Rules:
- Always start every prompt with: "children's book illustration, soft watercolor style, warm pastel colors, gentle lighting"
- Describe the scene visually: characters, setting, action, mood
- Keep character descriptions consistent across all 4 prompts
- Do NOT include text, speech bubbles, or letters in the image description
- Maximum 60 words per prompt

Output ONLY a JSON array, no other text:
[
  {
    "scene_ko": "<장면 제목 (한국어)>",
    "prompt": "<English image prompt>"
  }
]"""

SCENE_USER_TEMPLATE = """[동화 기획]
{plan}

[동화 전문]
{story}

이 동화의 핵심 장면 4개를 골라 이미지 프롬프트로 변환하세요."""


class FairyTaleImageGenerator:
    """Solar Pro + Pollinations.ai 기반 동화 삽화 생성기."""

    def __init__(self, api_key: str, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.output_dir = output_dir
        self.solar_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        os.makedirs(output_dir, exist_ok=True)

    # ── Solar: 장면 프롬프트 추출 ─────────────────────────────────────────────
    def _get_scene_prompts(self, story: str, plan: str) -> List[Dict]:
        logger.info("Solar Pro — 장면 프롬프트 추출 중...")
        payload = {
            "model": SOLAR_MODEL,
            "messages": [
                {"role": "system", "content": SCENE_SYSTEM},
                {"role": "user",   "content": SCENE_USER_TEMPLATE.format(plan=plan, story=story)},
            ],
            "max_tokens": 1024,
            "temperature": 0.5,
        }
        resp = requests.post(SOLAR_API_URL, headers=self.solar_headers, json=payload, timeout=60)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        scenes = json.loads(cleaned)
        logger.info(f"장면 {len(scenes)}개 추출 완료")
        return scenes

    # ── Pollinations.ai 이미지 생성 ───────────────────────────────────────────
    def _generate_image(self, prompt: str, width: int = 768, height: int = 768) -> Tuple[bytes, str]:
        encoded = urllib.parse.quote(prompt)
        url = f"{POLLINATIONS_URL.format(prompt=encoded)}?width={width}&height={height}&nologo=false&model=flux"
        logger.info(f"Pollinations.ai 요청 중...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        return resp.content, url

    # ── 메인 ─────────────────────────────────────────────────────────────────
    def generate(self, story: str, plan: str) -> Tuple[List[str], List[Dict]]:
        """
        동화 → 장면 추출 → 이미지 생성 → 저장.

        Returns:
            paths  : 저장된 PNG 파일 경로 목록 (또는 Pollinations URL)
            scenes : 장면별 프롬프트 정보 목록
        """
        scenes = self._get_scene_prompts(story, plan)

        paths = []
        for i, scene in enumerate(scenes, 1):
            prompt   = scene["prompt"]
            title_ko = scene.get("scene_ko", f"장면 {i}")
            save_path = os.path.join(self.output_dir, f"scene_{i:02d}.png")

            print(f"\n  🖼  [{i}/{len(scenes)}] {title_ko}")
            print(f"      → {prompt}")

            try:
                image_bytes, poll_url = self._generate_image(prompt)
                with open(save_path, "wb") as f:
                    f.write(image_bytes)
                logger.info(f"저장 완료: {save_path}")
                paths.append(save_path)
                print(f"      ✅ 저장: {save_path}")
                print(f"      🔗 URL: {poll_url}")
            except requests.HTTPError as e:
                logger.error(f"Pollinations.ai 오류 ({e.response.status_code}): {e}")
                print(f"      ⚠ 이미지 생성 실패: {e}")
                paths.append("")

        return paths, scenes
