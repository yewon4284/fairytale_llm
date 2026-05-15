"""
safeguard.py
카나나 세이프가드 8B를 이용한 1차 문장 단위 유해성 분류.

동작 방식:
- 동화를 문장 단위로 분리한다.
- 각 문장에 대해 safeguard를 호출하여 SAFE / UNSAFE-Sx 를 판정한다.
- UNSAFE 문장은 카테고리와 함께 "요주의 문장"으로 태깅하여 반환한다.
- 1차 평가의 역할은 최종 판단이 아닌 의심 문장 탐지 + 카테고리 태깅에 한정된다.

카테고리:
  S1 증오, S2 괴롭힘, S3 성적콘텐츠, S4 범죄,
  S5 아동성착취, S6 자살·자해, S7 잘못된정보
"""

import re
import logging
from typing import List, Tuple, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

# ── 상수 ──────────────────────────────────────────────────────────────────────
SAFEGUARD_MODEL_ID = "kakaocorp/kanana-safeguard-8b"

CATEGORY_DESC: Dict[str, str] = {
    "S1": "증오(차별·혐오)",
    "S2": "괴롭힘",
    "S3": "성적 콘텐츠",
    "S4": "범죄·폭력",
    "S5": "아동 성착취",
    "S6": "자살·자해",
    "S7": "잘못된 정보",
}

# 문장 분리용 정규식 (마침표, 물음표, 느낌표 뒤 공백 기준)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?요])\s+")

# kanana-safeguard 프롬프트 형식 (모델 카드 기준)
SAFEGUARD_PROMPT_TEMPLATE = (
    "<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
)


class KananaSafeguard:
    """카나나 세이프가드 8B 래퍼"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        logger.info(f"Safeguard 디바이스: {self.device}")
        self._load_model()

    def _load_model(self):
        logger.info(f"Safeguard 모델 로딩: {SAFEGUARD_MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(SAFEGUARD_MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            SAFEGUARD_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
        logger.info("Safeguard 모델 로딩 완료")

    def classify_sentence(self, sentence: str) -> str:
        """
        단일 문장을 분류한다.
        Returns: '<SAFE>' 또는 '<UNSAFE-Sx>'
        """
        prompt = SAFEGUARD_PROMPT_TEMPLATE.format(text=sentence)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 프롬프트 이후 생성 토큰만 디코딩
        generated = output[0][inputs["input_ids"].shape[-1]:]
        result = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return result

    def evaluate_story(self, story: str) -> Tuple[List[str], List[Dict]]:
        """
        동화 전체를 문장 단위로 평가한다.

        Returns:
            sentences      : 분리된 문장 리스트
            flagged        : 요주의 문장 정보 리스트
                            [{"idx": int, "sentence": str, "category": str, "desc": str}]
        """
        # 문장 분리 (빈 문장 제거)
        sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(story) if s.strip()]
        flagged: List[Dict] = []

        logger.info(f"1차 평가 시작 — 총 {len(sentences)}개 문장")

        for idx, sent in enumerate(sentences):
            result = self.classify_sentence(sent)
            logger.debug(f"  [{idx+1}] {result} | {sent[:40]}...")

            if result.startswith("<UNSAFE"):
                # '<UNSAFE-S4>' → 'S4'
                category = result.strip("<>").replace("UNSAFE-", "")
                desc = CATEGORY_DESC.get(category, "기타")
                flagged.append({
                    "idx": idx,
                    "sentence": sent,
                    "category": category,
                    "desc": desc,
                })
                logger.info(f"  ⚠ 요주의: [{idx+1}] {category}({desc}) — {sent[:50]}")

        logger.info(
            f"1차 평가 완료 — 요주의 문장 {len(flagged)}개 / 전체 {len(sentences)}개"
        )
        return sentences, flagged