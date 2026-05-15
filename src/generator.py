import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "kakaocorp/kanana-nano-2.1b-base"

class FairyTaleGenerator:
    def __init__(self):
        # 1. 장치 설정 (맥은 mps, 아니면 cuda, 그것도 아니면 cpu)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        print(f"현재 사용 중인 디바이스: {self.device}")
        print("카나나 nano 모델 로딩 중...")

        # 2. 모델 로딩 (device 설정을 self.device로 변경)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            padding_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        print("카나나 nano 모델 로딩 완료!")

    # src/generator.py

    def get_structured_guide(self, raw_topic: str) -> str:
        """
        [1단계] 카나나 Nano를 사용하여 주제를 구조화된 설계도로 만듭니다.
        """
        # 나노가 이해하기 쉽게 간단하고 명확한 지시를 줍니다.
        struct_prompt = f"""당신은 동화의 뼈대를 만드는 설계자입니다. 
사용자의 요청을 분석하여 등장인물과 사건 순서(동사 체인)를 짧게 요약하세요.

[규칙]
1. 인물 성별을 언급하지 마세요. (예: 활발한 아이, 차분한 친구)
2. 부모님이나 외부 개입 없이 아이들이 스스로 해결하는 구조로 만드세요.
3. 사건은 (행동1) -> (행동2) -> (해결) 순서로 아주 짧게 쓰세요.

주제: {raw_topic}

설계도:"""

        inputs = self.tokenizer(struct_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=200, # 설계도는 짧아야 하므로 제한
                temperature=0.3,    # 논리적이어야 하므로 낮게 설정
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        guide = self.tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return guide.strip()

    def generate(self, structured_guide: str, max_new_tokens: int = 1024) -> str:
        """
        솔라가 작성한 [동화 설계도]를 바탕으로 동화 본문을 생성합니다.
        (max_new_tokens를 1024로 늘려 충분한 분량을 확보합니다.)
        """
        
        # [수정 포인트] 프롬프트를 설계도 중심(Constrained Generation)으로 변경
        prompt = f"""당신은 6~7세 아동을 위한 전문 동화 작가입니다. 
아래 [동화 설계도]를 바탕으로, 다음 규칙을 엄격히 지쳐 오직 **아이들이 듣는 동화 본문**만 작성하세요.

[동화 설계도]
{structured_guide}

[집필 절대 규칙 - 위반 시 감점]
1. **설명 금지**: "이 동화는~", "[동화 본문]", "[동화 해설]" 같은 제목이나 설명은 절대 쓰지 마세요.
2. **즉시 시작**: 첫 문장은 반드시 주인공의 이름이나 "옛날에~", "어느 날~" 같은 이야기의 시작으로 시작하세요.
3. **해설 금지**: 이야기 끝에 교훈이나 해설을 덧붙이지 마세요. 이야기가 끝나면 그대로 종료하세요.
4. **언어**: 반드시 한국어(Korean)로만 작성하세요.
5. **분량**: 아이들이 지루하지 않게 10~15문장 내외로 풍성하게 묘사하세요.
6. **성별 중립성**: 인물에게 '남자아이', '여자아이' 혹은 성별을 암시하는 수식어(예: 씩씩한 소년, 얌전한 소녀)를 절대 사용하지 마세요. 오직 이름(루카, 카나)이나 '아이들', '친구'로만 지칭하세요.
7. **사건의 단방향 전개**: 이야기는 '기-승-전-결'의 순서로 단 한 번만 진행됩니다. 이미 일어난 사건(사과, 화해 등)을 뒤에서 반복하지 마세요.
8. **묘사의 경제성**: "이 동화는~" 같은 설명이나 마지막의 해설을 모두 삭제하고 오직 이야기 본문만 출력하세요.
9. **감정의 배치**: 미안함이나 두근거림 같은 감정 묘사는 사건(갈등)이 일어난 직후에 배치하여 인과관계를 명확히 하세요.

동화 시작:"""

        # 1. 토크나이징 및 장치 전송
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # 2. 모델 생성 설정 (안정적인 서사를 위해 temperature 조절)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,      # 약간의 창의성을 주되
                top_p=0.9,            # 논리적 일관성을 위해 top_p 조절
                repetition_penalty=1.1, # 문맥 반복 방지
            )

        generated_ids = output[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# 단독 테스트용
if __name__ == "__main__":
    gen = FairyTaleGenerator()
    topic = "유괴 예방에 도움이 되는 동화"
    result = gen.generate(topic)
    print("=" * 50)
    print("생성된 동화:")
    print("=" * 50)
    print(result)