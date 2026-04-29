import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "kakaocorp/kanana-safeguard-8b"

# 안전 카테고리 정의
UNSAFE_CATEGORIES = {
    "S1": "증오 (차별적 발화)",
    "S2": "괴롭힘",
    "S3": "성적 콘텐츠",
    "S4": "범죄",
    "S5": "아동 성착취",
    "S6": "자살·자해",
    "S7": "잘못된 정보",
}

class SafeguardEvaluator:
    def __init__(self):
        print("카나나 세이프가드 모델 로딩 중...")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("카나나 세이프가드 모델 로딩 완료!")

    def classify_sentence(self, sentence: str) -> dict:
        """
        단일 문장을 분류합니다.
        
        Args:
            sentence: 평가할 문장
            
        Returns:
            {
                "sentence": 원본 문장,
                "label": "SAFE" 또는 "UNSAFE-Sx",
                "is_unsafe": bool,
                "category": 카테고리 코드 (UNSAFE일 경우),
                "category_name": 카테고리 이름 (UNSAFE일 경우)
            }
        """
        messages = [
            {"role": "user", "content": sentence},
            {"role": "assistant", "content": ""}
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt"
        ).to(self.model.device)

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        gen_idx = input_ids.shape[-1]
        label = self.tokenizer.decode(output_ids[0][gen_idx], skip_special_tokens=True)

        is_unsafe = label.startswith("<UNSAFE")
        category = None
        category_name = None

        if is_unsafe:
            # "<UNSAFE-S4>" → "S4" 추출
            category = label.replace("<UNSAFE-", "").replace(">", "").strip()
            category_name = UNSAFE_CATEGORIES.get(category, "알 수 없음")

        return {
            "sentence": sentence,
            "label": label,
            "is_unsafe": is_unsafe,
            "category": category,
            "category_name": category_name,
        }

    def evaluate_story(self, story: str) -> dict:
        """
        동화 전체를 문장 단위로 분리하여 1차 평가합니다.
        
        Args:
            story: 평가할 동화 전체 텍스트
            
        Returns:
            {
                "sentences": 전체 문장 결과 리스트,
                "flagged_sentences": 요주의 문장 리스트,
                "is_all_safe": 전체 안전 여부
            }
        """
        # 문장 분리 (마침표, 느낌표, 물음표 기준)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', story.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        results = []
        flagged = []

        for sent in sentences:
            result = self.classify_sentence(sent)
            results.append(result)
            if result["is_unsafe"]:
                flagged.append(result)
                print(f"  ⚠️  요주의: [{result['category']}] {sent}")
            else:
                print(f"  ✅ 안전: {sent[:30]}...")

        return {
            "sentences": results,
            "flagged_sentences": flagged,
            "is_all_safe": len(flagged) == 0,
        }


# 단독 테스트용
if __name__ == "__main__":
    sg = SafeguardEvaluator()

    test_story = """
    토끼와 곰은 숲속 친구였어요.
    어느 날 곰이 토끼의 당근을 몰래 훔쳐 먹었어요.
    토끼는 매우 슬펐지만 곰에게 화를 내지 않고 왜 그랬는지 물어보았어요.
    곰은 배가 너무 고팠다고 사실대로 말했어요.
    토끼는 곰을 용서하고 함께 당근을 나눠 먹었어요.
    그 후로 둘은 더 좋은 친구가 되었답니다.
    """

    print("=" * 50)
    print("1차 평가 결과:")
    print("=" * 50)
    result = sg.evaluate_story(test_story)
    print(f"\n요주의 문장 수: {len(result['flagged_sentences'])}")
    print(f"전체 안전 여부: {result['is_all_safe']}")