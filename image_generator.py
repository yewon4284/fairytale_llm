"""
image_generator.py
동화 최종본에서 핵심 장면을 추출하고 Stable Diffusion XL로 이미지를 생성한다.

[흐름]
  1. Solar Pro — 동화(한국어)에서 핵심 장면 4개 추출 + 영어 SD 프롬프트 생성
  2. SDXL (로컬) — 장면별 이미지 생성
  3. outputs/ 디렉터리에 PNG 저장

[사전 설치]
  pip install "diffusers==0.30.3" accelerate
"""

import json
import logging
import os
import re
from typing import Dict, List, Tuple

import base64

import requests

logger = logging.getLogger(__name__)

SOLAR_API_URL  = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL    = "solar-pro"
SD_MODEL_ID    = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_OUTPUT_DIR = "outputs"
IMGUR_UPLOAD_URL = "https://api.imgur.com/3/image"


# ── 장면 추출 프롬프트 ────────────────────────────────────────────────────────
SCENE_SYSTEM = """You are a children's book art director.
Extract exactly 4 key scenes from a Korean fairy tale and write a Stable Diffusion image prompt for each.

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
    "prompt": "<English SD prompt>",
    "negative_prompt": "text, watermark, realistic, photo, dark, scary, violence, speech bubble, letters"
  }
]"""

SCENE_USER_TEMPLATE = """[동화 기획]
{plan}

[동화 전문]
{story}

이 동화의 핵심 장면 4개를 골라 SD 프롬프트로 변환하세요."""


class FairyTaleImageGenerator:
    """Solar Pro + Stable Diffusion XL(로컬) 기반 동화 삽화 생성기."""

    def __init__(self, api_key: str, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.output_dir = output_dir
        self.solar_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.pipe = None
        self.imgur_client_id = os.getenv("IMGUR_CLIENT_ID", "")
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

    # ── SDXL 파이프라인 로드 (지연 로딩) ─────────────────────────────────────
    def _load_pipeline(self):
        if self.pipe is not None:
            return
        try:
            import torch
            from diffusers import StableDiffusionXLPipeline
        except ImportError:
            raise RuntimeError(
                "diffusers 패키지가 필요합니다:\n"
                '  pip install "diffusers==0.30.3" accelerate'
            )
        except Exception as e:
            raise RuntimeError(
                "diffusers 로딩 실패. 아래 명령어로 버전을 맞춰주세요:\n"
                '  pip install "diffusers==0.30.3" accelerate\n'
                f"원본 오류: {e}"
            )

        logger.info(f"Stable Diffusion XL 로딩 중: {SD_MODEL_ID}")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")
        self.pipe.enable_attention_slicing()
        logger.info("Stable Diffusion XL 로딩 완료")

    # ── Imgur 업로드 ──────────────────────────────────────────────────────────
    def _upload_to_imgur(self, path: str) -> str:
        if not self.imgur_client_id:
            return path
        with open(path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        resp = requests.post(
            IMGUR_UPLOAD_URL,
            headers={"Authorization": f"Client-ID {self.imgur_client_id}"},
            data={"image": image_b64, "type": "base64"},
            timeout=30,
        )
        resp.raise_for_status()
        url = resp.json()["data"]["link"]
        logger.info(f"Imgur 업로드 완료: {url}")
        return url

    # ── 메인 ─────────────────────────────────────────────────────────────────
    def generate(self, story: str, plan: str) -> Tuple[List[str], List[Dict]]:
        """
        동화 → 장면 추출 → 이미지 생성 → 저장.

        Returns:
            paths  : 저장된 PNG 파일 경로 목록
            scenes : 장면별 프롬프트 정보 목록
        """
        scenes = self._get_scene_prompts(story, plan)
        self._load_pipeline()

        import torch
        paths = []
        for i, scene in enumerate(scenes, 1):
            prompt   = scene["prompt"]
            negative = scene.get("negative_prompt", "text, watermark, realistic, photo, dark, scary")
            title_ko = scene.get("scene_ko", f"장면 {i}")
            path     = os.path.join(self.output_dir, f"scene_{i:02d}.png")

            print(f"\n  🖼  [{i}/{len(scenes)}] {title_ko}")
            print(f"      → {prompt}")

            logger.info(f"이미지 생성 중 ({i}/{len(scenes)})")
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative,
                width=768,
                height=768,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]

            image.save(path)
            logger.info(f"저장 완료: {path}")

            url = self._upload_to_imgur(path)
            paths.append(url if url != path else path)

        return paths, scenes
