"""
image_generator.py
동화 핵심 장면을 추출하고 Pollinations.ai API로 이미지를 생성한다.

[흐름]
  1. Solar Pro — 동화에서 핵심 장면 4개 추출 + 영어 프롬프트 생성
  2. Pollinations.ai — 무료 공개 API로 이미지 생성 (GPU 불필요)
  3. outputs/ 디렉터리에 PNG 저장

[Pollinations.ai]
  - API 키 불필요, 무료 공개 API
  - 내부적으로 Flux 모델 사용
  - URL: https://image.pollinations.ai/prompt/{프롬프트}
"""

import json
import logging
import os
import re
import time
import urllib.parse
from typing import Dict, List, Tuple

import requests

logger = logging.getLogger(__name__)

SOLAR_API_URL      = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL        = "solar-pro"
POLLINATIONS_URL   = "https://image.pollinations.ai/prompt/{prompt}"
DEFAULT_OUTPUT_DIR = "outputs"
IMAGE_WIDTH        = 768
IMAGE_HEIGHT       = 768


# ── 장면 추출 프롬프트 ────────────────────────────────────────────────────────
SCENE_SYSTEM = """You are a children's book art director.
Extract exactly 4 key scenes from a Korean fairy tale and write a Pollinations/Flux image prompt for each.

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
        self.api_key    = api_key
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
        resp = requests.post(
            SOLAR_API_URL,
            headers=self.solar_headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        raw     = resp.json()["choices"][0]["message"]["content"].strip()
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        scenes  = json.loads(cleaned)
        logger.info(f"장면 {len(scenes)}개 추출 완료")
        return scenes

    # ── Pollinations: 이미지 생성 ─────────────────────────────────────────────
    def _generate_image(self, prompt: str, path: str, seed: int = 42) -> bool:
        """
        Pollinations.ai API로 이미지를 생성하고 저장한다.

        Returns:
            성공 여부 (bool)
        """
        encoded = urllib.parse.quote(prompt)
        url     = (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width={IMAGE_WIDTH}&height={IMAGE_HEIGHT}&seed={seed}&nologo=true"
        )
        logger.info(f"이미지 다운로드 중: {path}")
        logger.debug(f"URL: {url}")

        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=120)
                if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image"):
                    with open(path, "wb") as f:
                        f.write(resp.content)
                    logger.info(f"저장 완료: {path}")
                    return True
                else:
                    logger.warning(f"이미지 응답 이상 (시도 {attempt+1}): status={resp.status_code}")
                    time.sleep(3)
            except requests.RequestException as e:
                logger.warning(f"이미지 요청 실패 (시도 {attempt+1}): {e}")
                time.sleep(3)

        logger.error(f"이미지 생성 최종 실패: {path}")
        return False

    # ── 메인 ─────────────────────────────────────────────────────────────────
    def generate(self, story: str, plan: str) -> Tuple[List[str], List[Dict]]:
        """
        동화 → 장면 추출 → Pollinations 이미지 생성 → 저장.

        Returns:
            paths  : 저장된 PNG 파일 경로 목록
            scenes : 장면별 프롬프트 정보 목록
        """
        scenes = self._get_scene_prompts(story, plan)
        paths  = []

        for i, scene in enumerate(scenes, 1):
            prompt   = scene["prompt"]
            title_ko = scene.get("scene_ko", f"장면 {i}")
            path     = os.path.join(self.output_dir, f"scene_{i:02d}.png")

            print(f"\n  🖼  [{i}/{len(scenes)}] {title_ko}")
            print(f"      프롬프트: {prompt}")

            success = self._generate_image(prompt, path, seed=i * 10)
            if success:
                paths.append(path)
            else:
                logger.warning(f"장면 {i} 이미지 생성 실패 — 건너뜀")

        print(f"\n✅ 이미지 {len(paths)}장 생성 완료")
        return paths, scenes