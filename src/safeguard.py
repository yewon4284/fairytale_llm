"""
safeguard.py
카나나 세이프가드 8B로 동화 전체를 한 번에 유해성 분류합니다.

변경된 사용 방식:
  이전: 문장 단위 루프 → 속도 느리고 아동 동화는 항상 SAFE
  현재: 동화 전체 텍스트를 1회 입력 → 심각한 케이스(S5 아동 성착취 등)만 안전망으로 탐지

역할: 최종 판단이 아닌 심각한 유해 콘텐츠 조기 차단 안전망
      맥락 판단은 Solar(2차 평가)가 담당

카테고리:
  S1 증오  S2 괴롭힘  S3 성적 콘텐츠  S4 범죄
  S5 아동 성착취  S6 자살·자해  S7 잘못된 정보
"""

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class KananaSafeguard:
    def __init__(self, model_name: str = "kakaocorp/kanana-safeguard-8b"):
        print(f"[Safeguard] 모델 로딩 중: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
        print("[Safeguard] 모델 로딩 완료")

    def check(self, fairy_tale_text: str) -> dict:
        """
        동화 전체를 1회 입력하여 유해성을 분류합니다.

        Returns:
            {
                "is_safe": bool,
                "label": "SAFE" | "S1"~"S7",
                "category": str,   # 카테고리 설명 (SAFE면 빈 문자열)
            }
        """
        CATEGORIES = {
            "S1": "증오 (출신·인종·성별·종교 등 차별)",
            "S2": "괴롭힘 (위협·불쾌감·타인 공격 조장)",
            "S3": "성적 콘텐츠",
            "S4": "범죄 (폭력·불법 행위 기획)",
            "S5": "아동 성착취",
            "S6": "자살·자해",
            "S7": "잘못된 정보",
        }

        messages = [
            {"role": "user", "content": fairy_tale_text},
            {"role": "assistant", "content": ""},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt"
        ).to(self.model.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_idx = input_ids.shape[-1]
        result = self.tokenizer.decode(
            output_ids[0][gen_idx], skip_special_tokens=True
        ).strip()

        # 결과 파싱
        if "<SAFE>" in result or result.upper() == "SAFE":
            label = "SAFE"
        else:
            match = re.search(r"UNSAFE[-_]?(S[1-7])", result.upper())
            label = match.group(1) if match else "SAFE"

        is_safe = label == "SAFE"
        category = CATEGORIES.get(label, "") if not is_safe else ""

        if is_safe:
            print(f"[Safeguard] ✅ SAFE")
        else:
            print(f"[Safeguard] ❌ UNSAFE-{label}: {category}")

        return {"is_safe": is_safe, "label": label, "category": category}