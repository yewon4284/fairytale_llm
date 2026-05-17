"""
generator.py
카나나 로컬 모델을 이용해 동화 본문을 생성한다.

[지원 모델]
  KANANA_NANO  : kakaocorp/kanana-nano-2.1b-instruct   (2.1B, 경량/비교용)
  KANANA_15_8B : kakaocorp/kanana-1.5-8b-instruct-2505 (8B, 현재 기본값)

[역할 분리 원칙]
  기획(Plan)  → Solar Pro  (evaluator.py)
  동화 생성   → 카나나 1.5 8B (이 파일)
  1차 평가    → 카나나 세이프가드 8B (safeguard.py)
  2차 평가    → Solar Pro (evaluator.py)

[동화 길이 기준]
  6~7세 아동 그림책: 600~1,000자 (공백 제외)
  근거: 국립어린이청소년도서관 유아 그림책 기준(2019),
        Valentini et al.(2023) AoA<=6 어휘 연구 200~400 어절 권장
"""

import re
import logging
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


# ── 모델 ID 상수 ──────────────────────────────────────────────────────────────
KANANA_NANO  = "kakaocorp/kanana-nano-2.1b-instruct"
KANANA_15_8B = "kakaocorp/kanana-1.5-8b-instruct-2505"

DEFAULT_MODEL = KANANA_15_8B


# ── 편향 유발 금지어 ──────────────────────────────────────────────────────────
# 성별·인종·가족관계처럼 LLM 편향을 유발하는 단어를 입력 단계에서 차단한다.
# 단어 경계 체크로 "형광등", "오빠상" 등 합성어는 통과.
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
    if not stripped:
        return stripped
    # 한국어 문장 종결 부호: . ! ? 및 요/다/죠 등의 어미로 끝나는 경우 포함
    if stripped[-1] in ".!?":
        return stripped
    # 마지막 완성 문장 탐색
    m = re.search(r"(.*[.!?])", stripped, re.DOTALL)
    if m:
        trimmed = m.group(1).strip()
        logger.warning(
            f"문장 끝 잘림 감지 → 트리밍 적용 (원본 {len(stripped)}자 → {len(trimmed)}자)"
        )
        return trimmed
    return stripped


# ── 동화 기획 프롬프트 ───────────────────────────────────────────────────────
PLAN_SYSTEM = """당신은 아동 교육 전문가이자 동화 작가입니다.
사용자가 설명한 상황을 바탕으로, 6~7세 아동에게 교훈을 전달하는 동화를 기획하세요.

[전달 방식 옵션]
- 역지사지: 주인공이 피해자 입장을 직접 경험하며 깨닫는 방식
- 제3자 조언: 현명한 조력자(동물, 나무, 친구 등)가 주인공에게 가르침을 주는 방식
- 결과 체험: 잘못된 행동의 결과를 주인공이 직접 겪으며 반성하는 방식
- 감정 공감: 상대방의 감정을 느끼고 이해하는 과정을 통해 변화하는 방식

[제약 사항]
- 주인공 이름은 성별이 드러나지 않는 이름으로 설정 (예: 도담, 하늘, 솔이, 누리, 봄이)
- 공주/왕자/왕/여왕/아들/딸/남자아이/여자아이 사용 금지
- 특정 인종·직업·지역 고정관념 없이 설정
- 신체 위험 행동을 긍정적으로 묘사하지 마세요

[출력 형식] 반드시 아래 형식 그대로 출력하세요:
[핵심 교훈] <한 문장으로 명확하게>
[전달 방식] <위 4가지 중 선택한 방식 + 이유 한 문장>
[주인공 설정] <이름(성별 중립) + 처한 상황 한 문장>
[조력자/계기] <변화를 이끄는 존재 또는 사건 한 문장>
[결말 방향] <어떤 깨달음이나 변화로 마무리되는지 한 문장>
[동화 분위기] <따뜻한 / 유쾌한 / 진지한 / 모험적인 중 선택>"""

PLAN_USER_TEMPLATE = "다음 상황에 맞는 6~7세용 동화를 기획해 주세요:\n\n{user_request}"


# ── 동화 생성 시스템 프롬프트 ─────────────────────────────────────────────────
STORY_SYSTEM = """당신은 6~7세 아동을 위한 한국어 동화 작가입니다.
주어진 기획서와 참고 동화를 바탕으로 동화 본문만 작성하세요. 기획 내용을 반복하거나 설명하지 마세요.

[작성 규칙]
- 대상 독자: 6~7세 한국 아동
- 길이: 반드시 전체 글자 수는 700자 이상 1,200자 이하로 작성하세요 (공백 제외).
- 짧고 쉬운 단어를 사용하세요 (초등 1학년 수준).
- 구조: 기승전결이 명확해야 함 (도입 → 갈등 → 반성 → 해결 순서).
- 충분한 장면 묘사, 대화, 감정 표현을 넣어 이야기를 풍성하게 써주세요.
- 갈등이 있더라도 반성·사과·화해 등 긍정적 결말로 마무리하세요.
- 이야기 속에서 교훈이 자연스럽게 드러나야 합니다 (직접 설교 금지).
- 특정 성별·인종·직업을 고정관념화하지 마세요.
- 폭력·갈등 묘사: 교훈을 위해 필요할 경우 허용하되, 반드시 반성·화해로 이어질 것.
- 신체 위험 행동(높은 곳에서 뛰어내리기, 날카로운 물건 다루기 등)을 긍정적으로 묘사하지 마세요.
- 등장 인물의 성별이 드러나지 않는 이름을 사용하세요 (예: 도담, 하늘, 솔이, 누리, 봄이).
- 동물이 주인공이라면 코코, 토토, 하루 등 중성적 이름을 사용하세요.
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
            # device_map="auto" 사용 시 레이어가 분산될 수 있으므로
            # 실제 첫 번째 파라미터의 device를 기준으로 input_ids를 이동
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

    def plan(self, user_request: str) -> str:
        """사용자 요청을 받아 동화 기획서를 생성한다."""
        logger.info(f"카나나({self.model_id.split('/')[-1]}) — 동화 기획 생성 중...")
        plan_text = self._chat(
            system=PLAN_SYSTEM,
            user=PLAN_USER_TEMPLATE.format(user_request=user_request),
            max_new_tokens=512,
        )
        logger.info("카나나 기획 완료")
        return plan_text

    def generate(
        self,
        plan: str,
        rewrite_hint: str = "",
        few_shot_examples: str = "",
    ) -> str:
        """
        Solar Pro가 작성한 기획서(plan)를 받아 동화 본문을 생성한다.

        Args:
            plan              : Solar Pro가 생성한 동화 기획서
            rewrite_hint      : 이전 평가 수정 지시사항. 빈 문자열이면 무시.
            few_shot_examples : 데이터셋에서 가져온 참고 동화 텍스트 (퓨샷)
        """
        few_shot_section = ""
        if few_shot_examples:
            few_shot_section = (
                "\n\n[참고 동화 — 아래 동화들의 문체·구조·길이를 참고하세요]\n"
                + few_shot_examples
            )

        hint_section = ""
        if rewrite_hint:
            hint_section = (
                "\n\n[이전 평가 피드백 — 반드시 반영하세요]\n"
                + rewrite_hint
            )

        user_prompt = (
            f"아래 기획서를 바탕으로 6~7세용 한국어 동화 본문을 작성하세요.\n\n"
            f"[동화 기획서]\n{plan}"
            f"{few_shot_section}"
            f"{hint_section}\n\n"
            f"지금 바로 동화 본문만 작성하세요. (700자 이상 1,200자 이하, 공백 제외)"
        )

        logger.info(f"카나나({self.model_id.split('/')[-1]}) — 동화 본문 생성 중...")
        story = self._chat(system=STORY_SYSTEM, user=user_prompt)
        story = _trim_to_last_sentence(story)
        char_count = len(story.replace(" ", ""))
        if char_count < 700:
            logger.warning(f"길이 미달 — {char_count}자 (최소 700자)")
        elif char_count > 1200:
            logger.warning(f"길이 초과 — {char_count}자 (최대 1,200자)")
        else:
            logger.info(f"생성 완료 — 글자 수(공백 제외): {char_count}자")
        return story