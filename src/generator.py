"""
generator.py
카나나 로컬 모델을 이용해 동화 본문을 생성한다.

[지원 모델 — 모델 ID 상수로 관리]
  KANANA_NANO  : kakaocorp/kanana-nano-2.1b-instruct
                 세이프가드(8B)와 합산 ~10B → A100 80GB 여유
  KANANA_15_8B : kakaocorp/kanana-1.5-8b-instruct-2505  ← 현재 기본값
                 세이프가드와 합산 ~36GB, A100 80GB 단일 GPU 운용 가능

[역할 분리 원칙]
  기획(Plan)  → Solar API (evaluator.py)
  동화 생성   → 카나나 (이 파일)
  1차 평가    → 카나나 세이프가드 8B (safeguard.py)
  2차 평가    → Solar API (evaluator.py)

[동화 길이 기준]
  6~7세 아동 그림책: 700~1,800자 (공백 제외)
  근거: 데이터셋 385편 분석 결과 (의사소통 중앙값 1,669자, 25th percentile 1,294자)
"""

import re
import logging
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


# ── 모델 ID 상수 ──────────────────────────────────────────────────────────────
KANANA_NANO  = "kakaocorp/kanana-nano-2.1b-instruct"           # 2.1B, 비교 실험용
KANANA_15_8B = "kakaocorp/kanana-1.5-8b-instruct-2505"         # 8B,  현재 기본값

DEFAULT_MODEL = KANANA_NANO


# ── 편향 유발 금지어 ──────────────────────────────────────────────────────────
BIASED_WORDS = {
    "아들", "딸", "공주", "왕자", "왕", "여왕",
    "남자아이", "여자아이", "남자", "여자",
    "오빠", "언니", "형", "누나",
    "흑인", "백인", "동양인", "서양인",
    "한국인", "미국인", "중국인", "일본인",
}

BIAS_PATTERN = re.compile(
    r"(?<![가-힣a-zA-Z])"
    + "(" + "|".join(re.escape(w) for w in sorted(BIASED_WORDS, key=len, reverse=True)) + ")"
    + r"(?![가-힣a-zA-Z])",
    re.UNICODE,
)


def check_bias(user_input: str) -> Optional[str]:
    """편향 단어가 포함되면 해당 단어를 반환한다. 없으면 None."""
    m = BIAS_PATTERN.search(user_input)
    return m.group() if m else None


def _trim_to_last_sentence(text: str) -> str:
    """토큰 한도로 문장이 중간에 잘린 경우 마지막 완성 문장까지만 남긴다."""
    stripped = text.strip()
    if not stripped or stripped[-1] in ".!?":
        return stripped
    m = re.search(r"(.*[.!?])", stripped, re.DOTALL)
    if m:
        trimmed = m.group(1).strip()
        logger.warning(f"문장 끝 잘림 감지 → 트리밍 적용 ({len(stripped)}자 → {len(trimmed)}자)")
        return trimmed
    return stripped


# ── 시스템 프롬프트 ───────────────────────────────────────────────────────────
STORY_SYSTEM = """당신은 6~7세 아동을 위한 한국어 동화 작가입니다.
주어진 기획서와 참고 동화를 바탕으로 동화 본문만 작성하세요. 기획 내용을 반복하거나 설명하지 마세요.

[작성 규칙]
- 대상 독자: 6~7세 한국 아동
- 길이: 반드시 전체 글자 수는 700자 이상 1,800자 이하로 작성하세요 (공백 제외).
- 짧고 쉬운 단어를 사용하세요 (초등 1학년 수준).
- 구조: 기승전결이 명확해야 함 (도입→갈등→반성→해결 순서)
- 충분한 장면 묘사, 대화, 감정 표현을 넣어 이야기를 풍성하게 써주세요.
- 갈등이 있더라도 반성·사과·화해 등 긍정적 결말로 마무리하세요. 이야기 속에서 교훈이 자연스럽게 드러나야 함 (직접 설교 금지)
- 특정 성별·인종·직업을 고정관념화하지 마세요.
- 폭력·갈등 묘사: 교훈을 위해 필요할 경우 허용하되, 반드시 반성·화해로 이어질 것
- 등장 인물의 성별이 드러나지 않는 이름을 사용하세요 (예: 도담, 하늘, 솔이, 누리) (동물들이 주인공이라면 코코, 토토 등).
- 공주, 왕자, 왕, 여왕, 아들, 딸, 남자아이, 여자아이는 절대 사용하지 마세요."""


# ── Generator 클래스 ─────────────────────────────────────────────────────────
class FairyTaleGenerator:
    """카나나 로컬 모델 기반 동화 본문 생성기."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: Optional[str] = None):
        self.model_id = model_id
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        logger.info(f"Generator 모델: {self.model_id} / 디바이스: {self.device}")
        self._load_model()

    def _load_model(self):
        logger.info(f"Generator 모델 로딩 중: {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device in ("cuda", "mps") else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"Generator 모델 로딩 완료: {self.model_id}")

    def _fallback_to_cpu(self):
        """
        OOM 발생 시 CPU로 전환한다.
        device_map="auto" 모델은 .to("cpu")가 불완전하므로
        기존 모델을 해제 후 CPU에서 재로딩한다.
        """
        logger.warning("메모리 부족 → CPU 재로딩합니다.")
        self.device = "cpu"
        del self.model
        torch.cuda.empty_cache()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            device_map=None,
        )
        self.model.eval()
        logger.info("Generator CPU 재로딩 완료")

    def _chat(self, system: str, user: str, max_new_tokens: int = 1536) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

        def _run() -> str:
            # device_map="auto" 시 레이어가 분산될 수 있으므로
            # 모델의 실제 첫 번째 파라미터 디바이스를 사용한다.
            actual_device = next(self.model.parameters()).device
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(actual_device)
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            generated = output_ids[0][input_ids.shape[-1]:]
            return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        try:
            return _run()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self._fallback_to_cpu()
                return _run()
            raise

    def generate(self, plan: str, rewrite_hint: str = "", few_shot_examples: str = "") -> str:
        """
        Solar가 작성한 기획서(plan)를 받아 동화 본문을 생성한다.

        Args:
            plan              : Solar가 생성한 동화 기획
            rewrite_hint      : 이전 평가 수정 지시사항. 빈 문자열이면 무시.
            few_shot_examples : 데이터셋에서 가져온 참고 동화 텍스트 (퓨샷)
        """
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
            f"지금 바로 동화 본문만 작성하세요. (700자 이상 1,800자 이하, 공백 제외)"
        )

        logger.info(f"카나나({self.model_id.split('/')[-1]}) — 동화 본문 생성 중...")
        story = self._chat(system=STORY_SYSTEM, user=user_prompt)
        story = _trim_to_last_sentence(story)
        char_count = len(story.replace(" ", ""))
        if char_count < 700:
            logger.warning(f"길이 미달 — {char_count}자 (최소 700자)")
        elif char_count > 1800:
            logger.warning(f"길이 초과 — {char_count}자 (최대 1,800자)")
        else:
            logger.info(f"생성 완료 — 글자 수(공백 제외): {char_count}자")
        return story