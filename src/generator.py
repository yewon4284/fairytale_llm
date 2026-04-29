import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "kakaocorp/kanana-nano-2.1b-base"

class FairyTaleGenerator:
    def __init__(self):
        print("카나나 nano 모델 로딩 중...")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            padding_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        print("카나나 nano 모델 로딩 완료!")

    def generate(self, topic: str, max_new_tokens: int = 512) -> str:
        """
        동화 주제를 입력받아 동화 초안을 생성합니다.
        
        Args:
            topic: 동화 주제 (예: "토끼와 거북이의 우정")
            max_new_tokens: 생성할 최대 토큰 수
            
        Returns:
            생성된 동화 텍스트
        """
        prompt = f"""다음 주제로 6~7세 아동을 위한 동화를 써주세요.
주제: {topic}

조건:
- 300~350어절 분량
- 짧고 쉬운 문장 사용
- 기승전결 구조
- 긍정적인 교훈 포함

동화:"""

        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        )["input_ids"].to("cuda")

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                repetition_penalty=1.2,
            )

        # 프롬프트 제외하고 생성된 부분만 디코딩
        generated = output[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


# 단독 테스트용
if __name__ == "__main__":
    gen = FairyTaleGenerator()
    topic = "욕심쟁이 곰과 나눔을 배우는 토끼"
    result = gen.generate(topic)
    print("=" * 50)
    print("생성된 동화:")
    print("=" * 50)
    print(result)