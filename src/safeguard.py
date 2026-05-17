"""
safeguard.py
카나나 세이프가드 8B를 이용한 1차 문장 단위 유해성 분류.

[공식 사용법 기준 구현]
  - torch_dtype=bfloat16 (공식 예시)
  - apply_chat_template 사용 (raw string 템플릿 X)
  - user: 사용자 요청(동화 생성 요청) / assistant: 동화 문장
  - max_new_tokens=1 (단일 토큰 분류)

[동작 방식]
  동화를 문장 단위로 분리 → 각 문장을 assistant 역할로 입력 →
  SAFE / UNSAFE-Sx 판정 → UNSAFE 문장 태깅 후 반환.
  최종 PASS/FAIL 판단은 Solar Pro 2차 평가가 담당한다.

[카테고리]
  S1 증오·차별, S2 괴롭힘, S3 성적 콘텐츠, S4 범죄·폭력,
  S5 아동 성착취,  S6 자살·자해,  S7 잘못된 정보
"""

import gc
import re
import logging
from typing import Dict, List, Optional, Tuple

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

# 한국어 문장 종결 기준
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# 세이프가드에 전달할 user 역할 고정 문구
# assistant 역할에 동화 문장이 들어가므로 user는 생성 맥락을 알려준다.
SAFEGUARD_USER_PROMPT = "6~7세 아동을 위한 한국어 동화 본문입니다. 아래 내용이 안전한지 평가해 주세요."


class KananaSafeguard:
    """카나나 세이프가드 8B 래퍼 — 공식 사용법 기준."""

    def __init__(self, device: Optional[str] = None):
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        logger.info(f"Safeguard 디바이스: {self.device}")
        self._load_model()

    def _load_model(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        logger.info(f"Safeguard 모델 로딩: {SAFEGUARD_MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(SAFEGUARD_MODEL_ID)

        # 공식 예시: bfloat16
        dtype = (
            torch.bfloat16 if self.device in ("cuda", "mps")
            else torch.float32
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            SAFEGUARD_MODEL_ID,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Safeguard 모델 로딩 완료")

    def _fallback_to_cpu(self):
        """
        OOM 발생 시 CPU로 전환한다.
        device_map="auto" 모델은 .to("cpu")가 불완전하므로
        기존 모델을 해제 후 CPU에서 재로딩한다.
        """
        logger.warning("Safeguard 메모리 부족 → CPU 재로딩합니다.")
        self.device = "cpu"
        del self.model
        torch.cuda.empty_cache()
        self.model = AutoModelForCausalLM.from_pretrained(
            SAFEGUARD_MODEL_ID,
            torch_dtype=torch.float32,
            device_map=None,
        )
        self.model.eval()
        logger.info("Safeguard CPU 재로딩 완료")

    def _make_input_ids(self, sentence: str) -> torch.Tensor:
        """
        공식 사용법: apply_chat_template으로 입력 구성.
        user: 동화 생성 맥락 고정 문구
        assistant: 평가할 동화 문장
        """
        messages = [
            {"role": "user",      "content": SAFEGUARD_USER_PROMPT},
            {"role": "assistant", "content": sentence},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
        )

    def classify_sentence(self, sentence: str) -> str:
        """
        단일 문장을 분류한다 (공식 예시 방식).

        Returns:
            '<SAFE>' 또는 '<UNSAFE-Sx>'
        """
        actual_device = next(self.model.parameters()).device
        input_ids = self._make_input_ids(sentence).to(actual_device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        def _run() -> str:
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1,          # 공식 예시: 1토큰
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            gen_idx = input_ids.shape[-1]
            return self.tokenizer.decode(
                output_ids[0][gen_idx], skip_special_tokens=True
            )

        try:
            return _run()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self._fallback_to_cpu()
                return _run()
            raise

    def classify_batch(self, sentences: List[str], batch_size: int = 16) -> List[str]:
        """
        여러 문장을 배치로 분류한다.

        apply_chat_template 결과물은 문장마다 길이가 달라
        left-padding이 필요하다.

        Args:
            sentences  : 분류할 문장 리스트
            batch_size : 한 번에 처리할 문장 수

        Returns:
            각 문장에 대한 분류 결과 리스트 ('<SAFE>' 또는 '<UNSAFE-Sx>')
        """
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        results: List[str] = []

        def _run_batch(batch: List[str]) -> List[str]:
            # 각 문장을 apply_chat_template으로 토큰화
            encoded = [
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "user",      "content": SAFEGUARD_USER_PROMPT},
                        {"role": "assistant", "content": s},
                    ],
                    tokenize=True,
                    return_tensors="pt",
                ).squeeze(0)           # (seq_len,)
                for s in batch
            ]

            # 가장 긴 시퀀스 기준으로 left-padding
            max_len = max(t.shape[0] for t in encoded)
            pad_id  = self.tokenizer.pad_token_id

            input_ids_list = []
            attn_mask_list = []
            for t in encoded:
                pad_len = max_len - t.shape[0]
                padded  = torch.cat([torch.full((pad_len,), pad_id), t])
                mask    = torch.cat([torch.zeros(pad_len), torch.ones(t.shape[0])]).long()
                input_ids_list.append(padded)
                attn_mask_list.append(mask)

            actual_device = next(self.model.parameters()).device
            input_ids  = torch.stack(input_ids_list).to(actual_device)
            attn_mask  = torch.stack(attn_mask_list).to(actual_device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=pad_id,
                )

            batch_results = []
            for j in range(len(batch)):
                gen_token = outputs[j][max_len]   # 생성된 1개 토큰
                decoded   = self.tokenizer.decode(gen_token, skip_special_tokens=True).strip()
                batch_results.append(decoded)
            return batch_results

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            try:
                batch_results = _run_batch(batch)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self._fallback_to_cpu()
                    batch_results = _run_batch(batch)
                else:
                    raise
            results.extend(batch_results)

        return results

    def evaluate_story(self, story: str) -> Tuple[List[str], List[Dict]]:
        """
        동화 전체를 문장 단위로 1차 평가한다 (배치 처리).

        Returns:
            sentences : 분리된 문장 리스트
            flagged   : 요주의 문장 정보 리스트
                [{"idx": int, "sentence": str, "category": str, "desc": str}]
        """
        sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(story) if s.strip()]
        flagged: List[Dict] = []

        logger.info(f"1차 평가 시작 — 총 {len(sentences)}개 문장 (배치 처리)")

        results = self.classify_batch(sentences)

        for idx, (sent, result) in enumerate(zip(sentences, results)):
            logger.debug(f"  [{idx + 1}] {result} | {sent[:40]}...")
            if result.startswith("<UNSAFE"):
                category = result.strip("<>").replace("UNSAFE-", "")
                desc = CATEGORY_DESC.get(category, "기타")
                flagged.append({
                    "idx":      idx,
                    "sentence": sent,
                    "category": category,
                    "desc":     desc,
                })
                logger.info(
                    f"  ⚠ 요주의: [{idx + 1}] {category}({desc}) — {sent[:50]}"
                )

        logger.info(
            f"1차 평가 완료 — 요주의 {len(flagged)}개 / 전체 {len(sentences)}개"
        )
        return sentences, flagged