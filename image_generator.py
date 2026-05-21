"""
image_generator.py
동화를 기승전결 4분할하고 Pollinations.ai(FLUX)로 이미지를 생성한다.

[흐름]
  1. Solar Pro — 동화에서 주인공 외양을 영어 구문으로 추출
  2. Solar Pro — 동화를 기승전결 구조로 4분할 + 영어 이미지 프롬프트 생성
             (주인공 외양 구문을 모든 프롬프트에 고정 삽입)
  3. Pollinations.ai (FLUX) — 이미지 생성
  4. outputs/ 디렉터리에 PNG 저장

[Pollinations.ai]
  - API 키 불필요, 완전 무료
  - 모델: flux
  - 속도 제한 시 자동 재시도 (최대 3회)
"""

import json
import logging
import os
import re
import time
import urllib.parse
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

SOLAR_API_URL     = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL       = "solar-pro"
POLLINATIONS_BASE = "https://image.pollinations.ai/prompt"
DEFAULT_OUTPUT_DIR = "outputs"
IMAGE_WIDTH  = 768
IMAGE_HEIGHT = 768
FIXED_SEED   = 42  # 모든 이미지에 동일 시드 → 스타일 일관성 강화


# ── 주인공 외양 추출 프롬프트 ─────────────────────────────────────────────────
CHARACTER_SYSTEM = """You are a children's book character designer.
Extract the MAIN character's consistent visual appearance from the Korean fairy tale.
Output ONLY a single English phrase (max 25 words) describing:
- age/size, hair color/style, eye color, clothing color and style
- any distinctive features (e.g. freckles, glasses, hat)
Example: "a small 6-year-old girl with short brown pigtails, big round eyes, wearing a red striped dress and white apron"
No extra text, no punctuation at the end."""


# ── 장면 분할 + 프롬프트 생성 프롬프트 ────────────────────────────────────────
SCENE_SYSTEM = """You are a children's book art director and story analyst.

A FIXED CHARACTER DESCRIPTION will be provided. You MUST copy it VERBATIM into every prompt.

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
- Then insert the FIXED CHARACTER DESCRIPTION exactly as given — word for word, every time
- Describe the scene: setting, action, emotion
- Keep background/environment consistent with the story's world
- Do NOT include text, speech bubbles, or letters in the image
- Maximum 70 words per prompt

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
    """Solar Pro + Pollinations.ai(FLUX) 기반 동화 삽화 생성기."""

    def __init__(self, api_key: str, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.api_key    = api_key
        self.output_dir = output_dir
        self.solar_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        os.makedirs(output_dir, exist_ok=True)

    # ── Solar: 주인공 외양 추출 ───────────────────────────────────────────────
    def _extract_character(self, story: str) -> str:
        """Solar Pro로 주인공의 외양을 영어 구문으로 추출한다."""
        logger.info("Solar Pro — 주인공 외양 추출 중...")
        payload = {
            "model": SOLAR_MODEL,
            "messages": [
                {"role": "system", "content": CHARACTER_SYSTEM},
                {"role": "user",   "content": story},
            ],
            "max_tokens": 80,
            "temperature": 0.1,
        }
        try:
            resp = requests.post(
                SOLAR_API_URL,
                headers=self.solar_headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            desc = resp.json()["choices"][0]["message"]["content"].strip()
            # 따옴표 제거
            desc = desc.strip('"\'')
            logger.info(f"주인공 외양: {desc}")
            return desc
        except Exception as e:
            logger.warning(f"캐릭터 추출 실패, 기본값 사용: {e}")
            return "a small child with a friendly expression, wearing simple colorful clothes"

    # ── Solar: 기승전결 4분할 + 프롬프트 생성 ────────────────────────────────
    def _get_page_prompts(
        self, story: str, plan: str, character_desc: str
    ) -> Tuple[List[Dict], List[str]]:
        """
        동화를 기승전결 4분할하고 각 섹션의 이미지 프롬프트를 생성한다.
        character_desc를 모든 프롬프트에 고정 삽입하여 주인공 외양을 일치시킨다.
        """
        paragraphs = [p.strip() for p in story.split("\n") if p.strip()]
        numbered   = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(paragraphs))

        user_content = (
            f"[FIXED CHARACTER DESCRIPTION — copy verbatim into every prompt]\n"
            f"{character_desc}\n\n"
            f"[Story Plan]\n{plan}\n\n"
            f"[Full Story — {len(paragraphs)} paragraphs, numbered]\n{numbered}\n\n"
            f"Split this story into exactly 4 narrative sections "
            f"(introduction / conflict / reflection / resolution) based on content, not equal length. "
            f"Assign paragraph indices to each section. "
            f"Then write one English image prompt per section. "
            f"Each prompt MUST contain the fixed character description above, word for word. "
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
            base_prompt = (
                "children's book illustration, soft watercolor style, "
                "warm pastel colors, gentle lighting"
            )
            scenes = [
                {
                    "page": i + 1,
                    "section": section_names[i],
                    "paragraphs": list(range(i * per + 1, (i + 1) * per + 1)),
                    "page_summary_ko": f"섹션 {i + 1}",
                    "prompt": (
                        f"{base_prompt}, {character_desc}, "
                        f"a heartwarming {section_names[i]} scene from a Korean fairy tale"
                    ),
                }
                for i in range(4)
            ]

        # character_desc가 누락된 프롬프트에 강제 삽입
        for scene in scenes:
            prompt = scene.get("prompt", "")
            if character_desc.lower()[:20] not in prompt.lower():
                logger.warning(
                    f"[page {scene.get('page')}] 프롬프트에 캐릭터 묘사 누락 → 강제 삽입"
                )
                scene["prompt"] = (
                    "children's book illustration, soft watercolor style, "
                    f"warm pastel colors, gentle lighting, {character_desc}, "
                    + re.sub(
                        r"children's book illustration.*?gentle lighting[,.]?\s*",
                        "",
                        prompt,
                        flags=re.IGNORECASE,
                    ).strip()
                )

        logger.info(f"4분할 완료: {[s.get('section', '') for s in scenes]}")
        return scenes, paragraphs

    # ── Pollinations.ai: 단일 이미지 생성 ────────────────────────────────────
    def _generate_one(self, prompt: str, path: str, seed: int = FIXED_SEED) -> bool:
        """Pollinations.ai FLUX 모델로 이미지 한 장을 생성한다."""
        encoded = urllib.parse.quote(prompt)
        url = (
            f"{POLLINATIONS_BASE}/{encoded}"
            f"?width={IMAGE_WIDTH}&height={IMAGE_HEIGHT}"
            f"&seed={seed}&nologo=true&model=flux"
        )

        for attempt in range(1, 4):
            try:
                resp = requests.get(url, timeout=120)
                content_type = resp.headers.get("content-type", "")
                if resp.status_code == 200 and content_type.startswith("image"):
                    with open(path, "wb") as f:
                        f.write(resp.content)
                    logger.info(f"저장 완료: {path}")
                    return True
                elif resp.status_code == 402:
                    wait = attempt * 10
                    logger.warning(f"요청 한도 초과 402 (시도 {attempt}/3) — {wait}초 대기")
                    time.sleep(wait)
                else:
                    logger.warning(
                        f"응답 이상 (시도 {attempt}/3): status={resp.status_code}"
                    )
                    time.sleep(5)
            except requests.RequestException as e:
                logger.warning(f"요청 실패 (시도 {attempt}/3): {e}")
                time.sleep(5)

        logger.error(f"이미지 생성 최종 실패: {path}")
        return False

    # ── 메인 ─────────────────────────────────────────────────────────────────
    def generate(
        self, story: str, plan: str, n_pages: int = 4
    ) -> Tuple[List[str], List[Dict]]:
        """
        동화 → 주인공 외양 추출 → 기승전결 4분할 → 이미지 생성 → 저장.

        Returns:
            paths     : 저장된 PNG 파일 경로 목록
            page_info : 페이지별 정보 (section, paragraphs, summary, image)
        """
        # ── Step 1: 주인공 외양 추출 ─────────────────────────────────────────
        character_desc = self._extract_character(story)
        print(f"\n  👤 주인공 외양: {character_desc}")

        # ── Step 2: 기승전결 4분할 + 프롬프트 생성 ───────────────────────────
        scenes, paragraphs = self._get_page_prompts(story, plan, character_desc)

        print(f"\n  📄 기승전결 4분할 결과:")
        for scene in scenes:
            paras = scene.get("paragraphs", [])
            print(
                f"     [{scene['page']}] {scene.get('section', '').upper()} "
                f"(문단 {paras}) — {scene.get('page_summary_ko', '')}"
            )

        # ── Step 3: 이미지 생성 ───────────────────────────────────────────────
        print(f"\n  🎨 Pollinations.ai FLUX — {len(scenes)}장 순차 생성 시작...")
        print(f"     (seed={FIXED_SEED} 고정으로 스타일 일관성 유지)\n")

        paths = []
        for i, scene in enumerate(scenes, 1):
            prompt   = scene["prompt"]
            path     = os.path.join(self.output_dir, f"scene_{i:02d}.png")
            title_ko = scene.get("page_summary_ko", f"페이지 {i}")

            print(f"  🖼  [{i}/{len(scenes)}] {title_ko}")
            logger.debug(f"프롬프트: {prompt}")

            success = self._generate_one(prompt, path, seed=FIXED_SEED)
            if success:
                paths.append(path)
                print(f"         ✅ 완료")
            else:
                print(f"         ❌ 실패")

            if i < len(scenes):
                time.sleep(2)

        # ── Step 4: 결과 정리 ─────────────────────────────────────────────────
        page_info = [
            {
                "page":       s.get("page"),
                "section":    s.get("section"),
                "paragraphs": s.get("paragraphs", []),
                "summary_ko": s.get("page_summary_ko", ""),
                "prompt":     s.get("prompt", ""),
                "image":      os.path.basename(paths[i]) if i < len(paths) else None,
            }
            for i, s in enumerate(scenes)
        ]

        print(f"\n✅ 이미지 {len(paths)}/{len(scenes)}장 생성 완료")
        return paths, page_info