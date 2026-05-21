"""
image_generator.py
동화를 기승전결 4분할하고 HuggingFace Inference API로 이미지를 생성한다.

[흐름]
  1. Solar Pro — 동화를 기승전결 구조로 4분할 + 영어 이미지 프롬프트 생성
  2. HuggingFace Inference API — SDXL로 이미지 생성
  3. outputs/ 디렉터리에 PNG 저장

[HuggingFace Inference API]
  - HF_TOKEN 환경변수 필요
  - 모델: stabilityai/stable-diffusion-xl-base-1.0
  - 무료 티어: 속도 제한 있음 (큐 대기 가능)
"""

import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

SOLAR_API_URL  = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL    = "solar-pro"
POLLINATIONS_BASE = "https://image.pollinations.ai/prompt"
DEFAULT_OUTPUT_DIR = "outputs"
IMAGE_WIDTH    = 768
IMAGE_HEIGHT   = 768


# ── 장면 추출 프롬프트 ────────────────────────────────────────────────────────
SCENE_SYSTEM = """You are a children's book art director and story analyst.

Your job:
1. Read the Korean fairy tale carefully.
2. Split it into exactly 4 narrative sections based on story structure:
   - Section 1: Introduction (characters and setting introduced)
   - Section 2: Conflict (problem or tension arises)
   - Section 3: Reflection (character realizes mistake, turning point)
   - Section 4: Resolution (apology, reconciliation, happy ending)
3. For each section, identify which paragraphs belong to it.
4. Write one English image prompt per section that visually represents that section's key moment.

Image prompt rules:
- Always start with: "children's book illustration, soft watercolor style, warm pastel colors, gentle lighting"
- Describe the key scene: characters, setting, action, emotion
- Keep character appearance consistent across all 4 prompts
- Do NOT include text, speech bubbles, or letters
- Maximum 60 words per prompt

Output ONLY a JSON array with exactly 4 objects, no other text:
[
  {
    "page": 1,
    "section": "introduction",
    "paragraphs": [<1-based paragraph indices that belong to this section>],
    "page_summary_ko": "<이 섹션의 핵심 내용 한국어 한 줄>",
    "prompt": "<English image prompt>"
  }
]"""


class FairyTaleImageGenerator:
    """Solar Pro + HuggingFace Inference API 기반 동화 삽화 생성기."""

    def __init__(self, api_key: str, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.api_key    = api_key
        self.output_dir = output_dir
        self.solar_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        os.makedirs(output_dir, exist_ok=True)

    # ── Solar: 기승전결 4분할 + 프롬프트 생성 ────────────────────────────────
    def _get_page_prompts(self, story: str, plan: str) -> Tuple[List[Dict], List[str]]:
        paragraphs = [p.strip() for p in story.split('\n') if p.strip()]
        numbered   = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(paragraphs))

        user_content = (
            f"[Story Plan]\n{plan}\n\n"
            f"[Full Story — {len(paragraphs)} paragraphs, numbered]\n{numbered}\n\n"
            f"Split this story into exactly 4 narrative sections (introduction / conflict / "
            f"reflection / resolution) based on content, not equal length. "
            f"Assign paragraph indices to each section. "
            f"Then write one English image prompt per section. "
            f"Output a JSON array with exactly 4 objects."
        )

        logger.info("Solar Pro — 기승전결 4분할 + 이미지 프롬프트 생성 중...")
        payload = {
            "model": SOLAR_MODEL,
            "messages": [
                {"role": "system", "content": SCENE_SYSTEM},
                {"role": "user",   "content": user_content},
            ],
            "max_tokens": 1500,
            "temperature": 0.3,
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

        try:
            scenes = json.loads(cleaned)
            scenes = sorted(scenes, key=lambda s: s.get("page", 0))
        except json.JSONDecodeError as e:
            logger.error(f"프롬프트 JSON 파싱 실패: {e}\n{cleaned}")
            section_names = ["introduction", "conflict", "reflection", "resolution"]
            per = max(1, len(paragraphs) // 4)
            scenes = [
                {
                    "page": i + 1,
                    "section": section_names[i],
                    "paragraphs": list(range(i*per+1, (i+1)*per+1)),
                    "page_summary_ko": f"섹션 {i+1}",
                    "prompt": (
                        "children's book illustration, soft watercolor style, "
                        "warm pastel colors, gentle lighting, "
                        f"a heartwarming {section_names[i]} scene from a Korean fairy tale"
                    ),
                }
                for i in range(4)
            ]

        logger.info(f"4분할 완료: {[s.get('section','') for s in scenes]}")
        return scenes, paragraphs

    # ── HuggingFace: 단일 이미지 생성 ────────────────────────────────────────
    def _generate_one(self, prompt: str, path: str, seed: int = 42) -> bool:
        """Pollinations.ai로 이미지 한 장을 생성한다."""
        import urllib.parse
        encoded = urllib.parse.quote(prompt)
        url = (
            f"{POLLINATIONS_BASE}/{encoded}"
            f"?width={IMAGE_WIDTH}&height={IMAGE_HEIGHT}&seed={seed}&nologo=true&model=flux"
        )

        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=120)
                content_type = resp.headers.get("content-type", "")
                if resp.status_code == 200 and content_type.startswith("image"):
                    with open(path, "wb") as f:
                        f.write(resp.content)
                    logger.info(f"저장 완료: {path}")
                    return True
                elif resp.status_code == 402:
                    wait = (attempt + 1) * 10
                    logger.warning(f"요청 한도 초과 402 (시도 {attempt+1}/3) — {wait}초 대기")
                    time.sleep(wait)
                else:
                    logger.warning(f"응답 이상 (시도 {attempt+1}/3): status={resp.status_code}")
                    time.sleep(5)
            except requests.RequestException as e:
                logger.warning(f"요청 실패 (시도 {attempt+1}/3): {e}")
                time.sleep(5)

        logger.error(f"이미지 생성 최종 실패: {path}")
        return False

    # ── 메인 ─────────────────────────────────────────────────────────────────
    def generate(self, story: str, plan: str, n_pages: int = 4) -> Tuple[List[str], List[Dict]]:
        """
        동화 → 기승전결 4분할 → HF 이미지 생성 → 저장.

        Returns:
            paths     : 저장된 PNG 파일 경로 목록
            page_info : 페이지별 정보 (section, paragraphs, summary, image)
        """
        scenes, paragraphs = self._get_page_prompts(story, plan)

        print(f"\n  📄 기승전결 4분할 결과:")
        for scene in scenes:
            paras = scene.get("paragraphs", [])
            print(
                f"     [{scene['page']}] {scene.get('section','').upper()} "
                f"(문단 {paras}) — {scene.get('page_summary_ko','')}"
            )

        print(f"\n  🎨 HuggingFace SDXL — {len(scenes)}장 순차 생성 시작...")
        paths = []
        for i, scene in enumerate(scenes, 1):
            prompt   = scene["prompt"]
            path     = os.path.join(self.output_dir, f"scene_{i:02d}.png")
            title_ko = scene.get("page_summary_ko", f"페이지 {i}")

            print(f"  🖼  [{i}/{len(scenes)}] {title_ko}")
            logger.debug(f"프롬프트: {prompt}")

            success = self._generate_one(prompt, path, seed=i*10)
            if success:
                paths.append(path)
                print(f"         ✅ 완료")
            else:
                print(f"         ❌ 실패")

            if i < len(scenes):
                time.sleep(2)

        page_info = [
            {
                "page":       s.get("page"),
                "section":    s.get("section"),
                "paragraphs": s.get("paragraphs", []),
                "summary_ko": s.get("page_summary_ko", ""),
                "image":      os.path.basename(paths[i]) if i < len(paths) else None,
            }
            for i, s in enumerate(scenes)
        ]

        print(f"\n✅ 이미지 {len(paths)}/{len(scenes)}장 생성 완료")
        return paths, page_info