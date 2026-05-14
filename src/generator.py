"""
generator.py
카나나 모델로 한국어 동화를 생성합니다.

사용 모델:
  기본: kakaocorp/kanana-nano-2.1b-instruct  (경량, VRAM ~4GB)
  대안: kakaocorp/kanana-1.5-8b-instruct-2505 (고품질, VRAM ~16GB)
       → 변경 시 __init__의 model_name 기본값만 바꾸면 됩니다.

생성 기준 (6~7세):
  - 단어 수 150~350개 (srcWordEA 기준, 어절 수와 실용적으로 동일)
  - 문장당 평균 7단어 이하
  - 높임말 서술 (~했어요, ~였어요)
  - 편향 표현 없음
"""

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 6~7세 동화 적정 단어 수 (데이터셋 분석 + Valentini et al., 2023 기준)
# 6~7세 동화 적정 단어 수 (데이터셋 분석 + Valentini et al., 2023 기준)
TARGET_MIN_WORDS = 200
TARGET_MAX_WORDS = 350

SYSTEM_PROMPT = """당신은 6~7세 어린이를 위한 한국어 동화 작가입니다.

[창작 규칙]
1. 전체 길이: 200~350단어 (문장당 평균 7단어 이하)
2. 배경: 아이들이 공감할 수 있는 현대 일상 (유치원, 공원, 집 등). "옛날 옛적" 금지.
3. 등장인물: 주인공과 피해자를 중심으로. 제3자(친구, 어른 등)가 등장해도 되지만 조연 역할에 그쳐야 합니다.
4. 구성: 도입 → 갈등(나쁜 행동 발생) → 반성(스스로 느끼거나 누군가의 말을 듣고) → 사과 → 마무리
5. 갈등 해결 방식:
   - 나쁜 행동을 한 캐릭터가 반드시 직접 사과해야 합니다
   - 왜 사과하게 됐는지 계기(감정 변화)를 한 문장으로 명시해야 합니다
     예) 피해자가 우는 모습을 보고, 누군가의 말을 듣고, 혼자 생각하다가 등
   - 가해자가 직접 사과하고 피해자가 받아들이는 장면이 있어야 합니다
6. 갈등의 원인이 양쪽 모두에게 있을 때:
   - 각자 자신의 잘못을 인정하고 사과해야 합니다
   - 한쪽만 일방적으로 사과하고 끝내면 안 됩니다
   - 예) A가 B의 장난감을 뺏었지만 B도 먼저 예의 없이 행동했다면
         A는 뺏은 것을 사과하고, B는 예의 없이 굴었던 것을 사과해야 합니다
7. 아이들이 이해할 수 있는 쉬운 단어 사용
8. 편향 표현 금지:
   - 성별 명시 단어 사용 금지 (아들, 딸, 왕자, 공주 등)
   - 외모 중심 칭찬 금지
   - 직업 고정관념 금지
9. 서술문은 반드시 높임말: "~했어요", "~였어요"
   (대사 따옴표 안은 자유)
10. 동화 본문만 출력하세요.
    "총 단어 수", "교훈:", "이 동화는...", "좋은 동화였습니다", "아래는",
    번호 목록(1. 2. 3.) 같은 메타 설명은 절대 붙이지 마세요.
    동화를 한 번만 쓰세요. 완성되면 즉시 멈추세요."""


class FairyTaleGenerator:
    def __init__(
        self,
        model_name: str = "kakaocorp/kanana-nano-2.1b-instruct",
        # model_name: str = "kakaocorp/kanana-1.5-8b-instruct-2505",  # 고품질 대안
    ):
        print(f"[Generator] 모델 로딩 중: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
        print("[Generator] 모델 로딩 완료")

    def generate(
        self,
        user_request: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        few_shot_example: str = "",
    ) -> str:
        """
        동화를 생성합니다.
        few_shot_example: 데이터셋 동화 일부 (길이·문체 가이드용)
        """
        # few-shot 예시가 있으면 시스템 프롬프트에 추가
        system = SYSTEM_PROMPT
        if few_shot_example:
            system += f"""

[참고 동화 예시 - 길이와 문체를 참고하세요]
{few_shot_example}
---
위 예시와 비슷한 길이와 문체로 아래 요청에 맞는 동화를 써주세요."""

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_request},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][input_ids.shape[-1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        text = self.honorify(text)
        text = self.strip_meta(text)
        return text.strip()

    @staticmethod
    def strip_meta(text: str) -> str:
        """
        카나나가 동화 완성 후 덧붙이는 메타 텍스트를 제거합니다.
        동화 본문 마무리 문장은 건드리지 않습니다.
        """
        cut_patterns = [
            # "이 이야기는 ~교훈을 줍니다" 형태만 자름 (마무리 서술과 구분)
            r'\n+이\s*이야기는\s*.{0,30}(교훈|배울\s*수\s*있)',
            r'\n+이\s*이야기를\s*통해',
            r'\n+이\s*동화는\s*(친구|아이|어린이|서로|장난감).{0,20}(교훈|알려|줍니다)',
            r'\n+좋은\s*동화',
            r'\n+아래는\s*.{0,20}(입니다|제안)',
            r'\n+#{1,3}\s*(교훈|총\s*단어|Tip|moral)',
            r'\n+\d+\.\s+[가-힣]',
            r'\n+다음은\s*이\s*추가',
        ]
        result = text
        for pattern in cut_patterns:
            match = re.search(pattern, result, re.IGNORECASE)
            if match:
                result = result[:match.start()].rstrip()

        # "주인공인" 메타적 표현 제거
        result = re.sub(r'주인공인\s*', '', result)

        return result

    def count_words(self, text: str) -> int:
        """어절(공백 기준) 수를 반환합니다. 단어 수와 실용적으로 동일하게 취급합니다."""
        return len(text.split())

    def split_sentences(self, text: str) -> list[str]:
        """동화 텍스트를 문장 단위로 분리합니다."""
        sentences = re.split(r"(?<=[.!?])\s+|(?<=요\.)\s*\n|(?<=다\.)\s*\n", text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def honorify(text: str) -> str:
        """
        서술문의 반말 어미를 높임말로 후처리합니다.
        대사(따옴표 안)는 건드리지 않습니다.
        """
        quotes: list[str] = []

        def protect_quote(m: re.Match) -> str:
            quotes.append(m.group(0))
            return f"__Q{len(quotes) - 1}__"

        protected = re.sub(r'"[^"]*"', protect_quote, text)
        protected = re.sub(r"'[^']*'", protect_quote, protected)

        replacements = [
            (r'(았|었)어([.!?])', r'\1어요\2'),
            (r'([가-힣])어([.!?])', r'\1어요\2'),
            (r'([가-힣])지([.!?])', r'\1지요\2'),
            (r'(했|됐|났|갔|왔|봤|줬|뺐)다([.!?])', r'\1어요\2'),
            (r'(았|었)어(\s*\n)', r'\1어요\2'),
            (r'([가-힣])어(\s*\n)', r'\1어요\2'),
        ]
        for pattern, repl in replacements:
            protected = re.sub(pattern, repl, protected)

        for i, q in enumerate(quotes):
            protected = protected.replace(f"__Q{i}__", q)
        return protected