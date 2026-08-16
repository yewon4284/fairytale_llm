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
  S5 아동성착취(그루밍 포함), S6 자살·자해, S7 잘못된정보
"""

import re
import logging
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logger = logging.getLogger(__name__)

# ── 상수 ──────────────────────────────────────────────────────────────────────
SAFEGUARD_MODEL_ID = "kakaocorp/kanana-safeguard-8b"

CATEGORY_DESC: Dict[str, str] = {
    "S1": "증오(차별·혐오)",
    "S2": "괴롭힘",
    "S3": "성적 콘텐츠",
    "S4": "범죄·폭력",
    "S5": "아동 성착취(그루밍 포함)",
    "S6": "자살·자해",
    "S7": "잘못된 정보",
}

# 파인튜닝된 S5 어댑터 경로 (학습 후 설정)
S5_ADAPTER_PATH: str = "finetune/kanana-s5-adapter/final_adapter"

# 문장 분리용 정규식 (마침표, 물음표, 느낌표 뒤 공백 기준)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?요])\s+")

# kanana-safeguard 프롬프트 형식 (모델 카드 기준)
SAFEGUARD_PROMPT_TEMPLATE = (
    "<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
)

# 문맥 윈도우 분류용 헤더 (실험적 기능 — 문장 단독 분류 시 발생하는 도메인 오탐
# 완화 목적. 아동 정보책/전래동화에서 "엉덩이/똥/방귀" 같은 표현이 앞뒤 문맥 없이
# 그 문장만 보면 S3로 오인되는 사례가 실측으로 확인됨. 이 헤더로 모델에게 앞뒤
# 문장은 참고용이고 [분류대상]만 분류하라고 명시적으로 지시한다.
CONTEXT_PROMPT_HEADER = (
    "다음은 동화의 연속된 문장들입니다. [분류대상]으로 표시된 문장 하나만 "
    "SAFE 또는 UNSAFE-Sx로 분류하세요. 앞뒤 문장은 맥락 참고용이며 분류 대상이 아닙니다.\n\n"
)


class KananaSafeguard:
    """카나나 세이프가드 8B 래퍼 (S5 LoRA 어댑터 선택 지원)"""

    def __init__(self, device: str = "cuda", use_s5_adapter: bool = True):
        self.device = device
        self.use_s5_adapter = use_s5_adapter
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

        adapter_path = Path(S5_ADAPTER_PATH).resolve()
        has_weights = (
            (adapter_path / "adapter_model.safetensors").exists()
            or (adapter_path / "adapter_model.bin").exists()
        )
        if self.use_s5_adapter and has_weights:
            logger.info(f"S5 LoRA 어댑터 로딩: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
            logger.info("S5 어댑터 로딩 완료")
        elif self.use_s5_adapter:
            logger.warning(
                f"S5 어댑터 가중치 없음({adapter_path}). 베이스 모델로만 실행합니다. "
                "파인튜닝 후 S5_ADAPTER_PATH를 확인하세요."
            )

        self.model.eval()
        logger.info("Safeguard 모델 로딩 완료")

    @staticmethod
    def _build_context_content(sentence: str, prev_context: str, next_context: str) -> str:
        parts = [CONTEXT_PROMPT_HEADER]
        if prev_context:
            parts.append(f"[이전 문맥] {prev_context}\n")
        parts.append(f"[분류대상] {sentence}\n")
        if next_context:
            parts.append(f"[다음 문맥] {next_context}\n")
        return "".join(parts)

    def classify_sentence(self, sentence: str, prev_context: str = "", next_context: str = "") -> str:
        """
        단일 문장을 분류한다.

        prev_context/next_context를 주면(비어있지 않으면) 앞뒤 문맥을 포함한
        프롬프트로 분류한다 — 문장을 단독으로 볼 때 발생하는 도메인 오탐(예: 전래동화
        방귀 장면이 S3로 오인)을 완화하기 위한 실험적 기능. 베이스 모델이 이 학습 시
        보지 못한 프롬프트 형식이라 효과는 실측 검증이 필요함 (test_safeguard_context.py 참고).

        Returns: '<SAFE>' 또는 '<UNSAFE-Sx>'
        """
        device = next(self.model.parameters()).device
        if prev_context or next_context:
            content = self._build_context_content(sentence, prev_context, next_context)
        else:
            content = sentence
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True, return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 프롬프트 이후 생성 토큰만 디코딩
        generated = output[0][input_ids.shape[-1]:]
        result = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return result

    def evaluate_story(
        self, story: str, use_context: bool = False, context_window: int = 1
    ) -> Tuple[List[str], List[Dict]]:
        """
        동화 전체를 문장 단위로 평가한다.

        Args:
            use_context    : True면 각 문장을 분류할 때 앞뒤 context_window개 문장을
                             맥락으로 같이 넣는다 (기본 False — 기존 동작과 동일,
                             하위 호환 유지).

                             ⚠ 실측 결과(2026-08-16, test_safeguard_context.py, kanana_solar_eval
                             기준): 오탐(FP) 44건 중 82%가 해소됐지만, 정탐(TP, 진짜 위험한
                             S3/S5/S6) 57건 중 63.2%(36건)를 놓치는 심각한 재현율 손실이 함께
                             발생함 (특히 S3+S5 동시 태깅 사례에서 S3만 사라지는 패턴 다수 —
                             문맥이 서사/픽션 프레이밍으로 작용해 모델이 관대해지는 것으로 추정,
                             LLM에서 흔한 "이야기니까 괜찮다" 식 실패 패턴과 유사).
                             결론: 아동 안전 도메인에서 이 트레이드오프는 용납 불가 —
                             프로덕션 기본값을 True로 바꾸지 말 것. 오탐 완화는 evaluator.py의
                             구조화된 재검토 규칙(Solar 2차 평가)에 맡기고, 이 옵션은 진단/실험
                             용도로만 남겨둔다.
            context_window : use_context=True일 때 앞/뒤로 포함할 문장 수.

        Returns:
            sentences      : 분리된 문장 리스트
            flagged        : 요주의 문장 정보 리스트
                            [{"idx": int, "sentence": str, "category": str, "desc": str}]
        """
        # 문장 분리 (빈 문장 제거)
        sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(story) if s.strip()]
        flagged: List[Dict] = []

        logger.info(
            f"1차 평가 시작 — 총 {len(sentences)}개 문장"
            + (f" (문맥 윈도우 {context_window} 사용)" if use_context else "")
        )

        for idx, sent in enumerate(sentences):
            if use_context:
                prev_ctx = " ".join(sentences[max(0, idx - context_window):idx])
                next_ctx = " ".join(sentences[idx + 1:idx + 1 + context_window])
                result = self.classify_sentence(sent, prev_ctx, next_ctx)
            else:
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
                logger.info(f"  요주의: [{idx+1}] {category}({desc}) — {sent[:50]}")

        logger.info(
            f"1차 평가 완료 — 요주의 문장 {len(flagged)}개 / 전체 {len(sentences)}개"
        )
        return sentences, flagged