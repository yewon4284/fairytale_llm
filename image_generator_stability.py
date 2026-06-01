"""
image_generator.py
동화를 기승전결 4분할하고 Stability AI (Stable Image Core)로 이미지를 생성한다.

[흐름]
  1. Solar Pro — 주인공 외양을 영어 구문으로 추출
  2. Solar Pro — 동화를 기승전결 4분할 + 영어 이미지 프롬프트 생성
  3. Stability AI Stable Image Core — 이미지 생성
  4. outputs/ 디렉터리에 PNG 저장

[필요한 환경변수]
  SOLAR_API_KEY      : Upstage Solar API 키
  STABILITY_API_KEY  : Stability AI API 키
                       https://platform.stability.ai/account/keys 에서 발급
                       가입 시 25 크레딧 무료 (이미지 1장 = 3 크레딧)
"""

import json
import logging
import os
import re
import time
from typing import Dict, List, Tuple

import requests

logger = logging.getLogger(__name__)

SOLAR_API_URL      = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL        = "solar-pro"
STABILITY_API_URL  = "https://api.stability.ai/v2beta/stable-image/generate/core"
DEFAULT_OUTPUT_DIR = "outputs"
IMAGE_WIDTH        = 768
IMAGE_HEIGHT       = 768
FIXED_SEED         = 42


CHARACTER_SYSTEM = """You are a children's book character designer.
Extract the MAIN character's consistent visual appearance from the Korean fairy tale.
Output ONLY a single English phrase (max 25 words) describing:
- age/size, hair color/style, eye color, clothing color and style
- any distinctive features (e.g. freckles, glasses, hat)
Example: "a small 6-year-old girl with short brown pigtails, big round eyes, wearing a red striped dress and white apron"
No extra text, no punctuation at the end."""


SCENE_SYSTEM = """You are a children's book art director and story analyst.

A FIXED CHARACTER DESCRIPTION will be provided. You MUST copy it VERBATIM into every prompt.

Your job:
1. Read the Korean fairy tale carefully.
2. Split it into exactly 4 narrative sections:
   - Section 1: Introduction (characters and setting introduced)
   - Section 2: Conflict (problem or tension arises)
   - Section 3: Reflection (character realizes mistake, turning point)
   - Section 4: Resolution (apology, reconciliation, happy ending)
3. For each section write one English image prompt.

Image prompt rules:
- Always start with: "children's book illustration, soft watercolor style, warm pastel colors, gentle lighting"
- Then insert the FIXED CHARACTER DESCRIPTION exactly as given — word for word, every time
- Describe the SPECIFIC ACTION and LOCATION happening in this scene
- If two characters appear, describe BOTH and their interaction clearly
- Setting must match the story exactly: school path / playground / schoolyard / classroom
- Do NOT use autumn forest or unrelated nature scenes unless the story mentions them
- Do NOT include text, speech bubbles, or letters in the image
- Maximum 70 words per prompt

Output ONLY a JSON array with exactly 4 objects, no other text:
[
  {
    "page": 1,
    "section": "introduction",
    "paragraphs": [<1-based paragraph indices>],
    "page_summary_ko": "<이 섹션의 핵심 내용 한국어 한 줄>",
    "prompt": "<English image prompt>"
  }
]"""


class FairyTaleImageGenerator:

    def __init__(
        self,
        api_key: str,
        stability_key: str = "",
        output_dir: str = DEFAULT_OUTPUT_DIR,
    ):
        self.api_key       = api_key
        self.stability_key = stability_key or os.getenv("STABILITY_API_KEY", "")
        self.output_dir    = output_dir
        self.solar_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }
        os.makedirs(output_dir, exist_ok=True)

    # ── Solar: 주인공 외양 추출 ───────────────────────────────────────────────
    def _extract_character(self, story: str) -> str:
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
                SOLAR_API_URL, headers=self.solar_headers, json=payload, timeout=30
            )
            resp.raise_for_status()
            desc = resp.json()["choices"][0]["message"]["content"].strip().strip('"\'')
            logger.info(f"주인공 외양: {desc}")
            return desc
        except Exception as e:
            logger.warning(f"캐릭터 추출 실패, 기본값 사용: {e}")
            return "a small child with a friendly expression, wearing simple colorful clothes"

    # ── Solar: 기승전결 4분할 + 프롬프트 생성 ────────────────────────────────
    def _get_page_prompts(
        self, story: str, plan: str, character_desc: str
    ) -> Tuple[List[Dict], List[str]]:
        paragraphs = [p.strip() for p in story.split("\n") if p.strip()]
        numbered   = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(paragraphs))

        user_content = (
            f"[FIXED CHARACTER DESCRIPTION — copy verbatim into every prompt]\n"
            f"{character_desc}\n\n"
            f"[Story Plan]\n{plan}\n\n"
            f"[Full Story — {len(paragraphs)} paragraphs, numbered]\n{numbered}\n\n"
            f"Split into 4 narrative sections and write one image prompt per section.\n"
            f"Each prompt MUST:\n"
            f"1. Contain the fixed character description above, word for word\n"
            f"2. Describe the SPECIFIC action and EXACT location from those paragraphs\n"
            f"3. If two characters are present, describe both and their interaction clearly\n"
            f"4. Match the story setting precisely\n"
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
            SOLAR_API_URL, headers=self.solar_headers, json=payload, timeout=60
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
            per  = max(1, len(paragraphs) // 4)
            base = "children's book illustration, soft watercolor style, warm pastel colors, gentle lighting"
            scenes = [
                {
                    "page": i + 1,
                    "section": section_names[i],
                    "paragraphs": list(range(i * per + 1, (i + 1) * per + 1)),
                    "page_summary_ko": f"섹션 {i + 1}",
                    "prompt": f"{base}, {character_desc}, a heartwarming {section_names[i]} scene",
                }
                for i in range(4)
            ]

        # character_desc 누락 시 강제 삽입
        for scene in scenes:
            prompt = scene.get("prompt", "")
            if character_desc.lower()[:20] not in prompt.lower():
                logger.warning(f"[page {scene.get('page')}] 캐릭터 묘사 누락 → 강제 삽입")
                base = "children's book illustration, soft watercolor style, warm pastel colors, gentle lighting"
                rest = re.sub(
                    r"children's book illustration.*?gentle lighting[,.]?\s*",
                    "", prompt, flags=re.IGNORECASE
                ).strip()
                scene["prompt"] = f"{base}, {character_desc}, {rest}"

        logger.info(f"4분할 완료: {[s.get('section', '') for s in scenes]}")
        return scenes, paragraphs

    # ── Stability AI: 이미지 생성 ────────────────────────────────────────────
    def _generate_one(self, prompt: str, path: str, seed: int = FIXED_SEED) -> bool:
        """Stability AI Stable Image Core로 이미지 한 장을 생성한다."""
        for attempt in range(1, 4):
            try:
                resp = requests.post(
                    STABILITY_API_URL,
                    headers={
                        "authorization": f"Bearer {self.stability_key}",
                        "accept": "image/*",
                    },
                    files={"none": ""},
                    data={
                        "prompt":        prompt,
                        "output_format": "png",
                        "width":         IMAGE_WIDTH,
                        "height":        IMAGE_HEIGHT,
                        "seed":          seed,
                    },
                    timeout=120,
                )

                if resp.status_code == 200:
                    with open(path, "wb") as f:
                        f.write(resp.content)
                    logger.info(f"저장 완료: {path}")
                    return True

                elif resp.status_code == 402:
                    print("         ❌ 크레딧 부족 — https://platform.stability.ai 에서 충전하세요.")
                    return False

                elif resp.status_code == 429:
                    wait = attempt * 10
                    print(f"         ⏳ 요청 한도 초과 (시도 {attempt}/3) — {wait}초 대기")
                    time.sleep(wait)

                else:
                    try:
                        msg = resp.json()
                    except Exception:
                        msg = resp.text[:200]
                    logger.warning(f"응답 이상 (시도 {attempt}/3): {resp.status_code} / {msg}")
                    time.sleep(5)

            except requests.RequestException as e:
                logger.warning(f"요청 실패 (시도 {attempt}/3): {e}")
                time.sleep(5)

        logger.error(f"이미지 생성 최종 실패: {path}")
        return False

    # ── 메인 ─────────────────────────────────────────────────────────────────
    def generate(self, story: str, plan: str, n_pages: int = 4) -> Tuple[List[str], List[Dict]]:
        # 1. 주인공 외양 추출
        character_desc = self._extract_character(story)
        print(f"\n  👤 주인공 외양: {character_desc}")

        # 2. 기승전결 4분할 + 프롬프트
        scenes, paragraphs = self._get_page_prompts(story, plan, character_desc)

        print(f"\n  📄 기승전결 4분할 결과:")
        for scene in scenes:
            paras = scene.get("paragraphs", [])
            print(
                f"     [{scene['page']}] {scene.get('section','').upper()} "
                f"(문단 {paras}) — {scene.get('page_summary_ko','')}"
            )

        print(f"\n  🎨 Stability AI Stable Image Core — {len(scenes)}장 순차 생성 시작...\n")

        paths = []
        for i, scene in enumerate(scenes, 1):
            prompt   = scene["prompt"]
            path     = os.path.join(self.output_dir, f"scene_{i:02d}.png")
            title_ko = scene.get("page_summary_ko", f"페이지 {i}")
            print(f"  🖼  [{i}/{len(scenes)}] {title_ko}")
            print(f"       프롬프트: {prompt[:100]}...")
            success = self._generate_one(prompt, path, seed=FIXED_SEED)
            print(f"         {'✅ 완료' if success else '❌ 실패'}")
            if success:
                paths.append(path)
            if i < len(scenes):
                time.sleep(1)

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