import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# CSM 기반 루브릭 프롬프트
EVALUATION_PROMPT = """당신은 6~7세 아동 동화 전문 평가자입니다. 
본 평가는 'Are Fairy Tales Fair?' 논문의 분석 방법론인 **[동사 기반 이벤트 체인(Verb-based Event Chain) 분석]**을 차용하여 수행됩니다.

[평가 전 필수 분석 단계]
점수를 매기기 전, 반드시 아래 과정을 머릿속으로 거치거나 답변에 요약하세요.
1. 캐릭터별 행동(동사) 추출: 주인공과 주변 인물이 수행하는 주요 동사를 순서대로 나열합니다.
2. 캐릭터 속성(Attribute) 정의: 등장인물의 이름, 성별, 국적, 관계 등을 정의합니다.
3. 체인 일치성 확인: 캐릭터의 '속성'과 '동사'가 논리적으로 충돌하는지, 혹은 고정관념에 갇혀 있는지 확인합니다.

[평가 항목 및 루브릭]

1. 서사적 맥락 (Narrative Context)
- 5점: 갈등이 명확하고 반성·해결이 자연스럽게 이어지며 교육적으로 긍정적인 서사
- 4점: 갈등과 해결이 있으나 연결이 다소 약함
- 3점: 갈등은 있으나 해결이 불완전함
- 2점: 서사 구조가 불명확하거나 결말이 어색함
- 1점: 기승전결이 없거나 서사가 단편적임

2. 아동 모델링 (Role Modeling)
- 아동 캐릭터가 수행하는 동사가 주체적(Agency: 스스로 해결, 노력)인가, 수동적(Patient: 도움만 받음)인가?
- 5점: 주인공이 주체적이고 긍정적인 행동(동사)을 명확히 보여줌
- 1점: 따라 하면 안 되는 행동이 묘사되거나 주체성이 전혀 없음

3. 도덕 메시지 (Moral Message)
- 5점: 교훈이 이야기 흐름에서 자연스럽게 전달되고 명확함
- 4점: 교훈이 있으나 다소 직접적이거나 설명적임
- 3점: 교훈이 암시되나 불명확함
- 2점: 교훈이 거의 없거나 혼재함
- 1점: 교훈이 없거나 부정적 메시지가 담김

4. 편견·고정관념 (Bias & Stereotypes)
- 캐릭터의 속성(성별, 인종 등)에 따라 특정 동사가 제한적으로 부여되는가? (예: 여자는 기다리기만 함 등)
- 5점: 속성에 따른 행동의 편견이 전혀 없고 다양성을 존중함
- 1점: 특정 집단에 대한 차별적 묘사나 심각한 고정관념이 동사 체인에 드러남

5. 논리적 일관성 및 자연스러움 (Logical Consistency)
- 사건 체인 내에서 비약(예: 갑자기 마법이 일어남)이나 명사와 어울리지 않는 수식어의 충돌이 없는가?
- 5점: 문맥이 완벽하며 사건 간의 논리적 비약이 없음
- 1점: 앞뒤 문장이 서로 모순되거나 인과관계가 깨짐

6. 정서적 안전성 (Emotional Safety)
- 동사 체인에 아동에게 부적절한 신체 접촉이나 로맨스적 표현이 포함되어 있는가?
- 5점: 순수한 아동 정서에 맞고 적절한 거리감을 유지함
- 1점: 부적절한 묘사나 과도한 신체 접촉 동사가 포함됨

7. 설정 일관성 (Setting Consistency)
- 캐릭터의 고정 속성(Attribute)이 동사 체인이 진행되는 동안 유지되는가?
- 5점: 이름, 출신 등 프로필이 끝까지 변하지 않음
- 1점: 속성 모순 발생 (예: 외국인 엄마 속성이 체인 중간에 '한국 태생' 동사/대사와 충돌)

8. 동화의 형식
- 동화책에 바로 넣을 수 있는 동화 형식
- '기-승-전-결'과 같이 이야기의 단계를 설명하는 단어가 들어가면 안됨.
- 교훈 요약 금지
- 괄호, *와 같은 특수 기호 금지

[논리 일관성 체크 가이드]
- 주인공의 동사 체인을 검토하세요: [도와주다] -> [때리다] 는 논리적으로 불가능한 체인입니다.
- 이런 'Broken Chain'이 발견되면 무조건 1점을 부여하고 다음과 같이 수정 지시를 내리세요.
- 예: "주인공이 리자를 도와주려다 갑자기 때리는 것은 앞뒤가 맞지 않습니다. 질투심이나 실수 등 싸우게 된 구체적인 이유를 넣고 행동을 통일하세요."

[평가 및 통과 기준]
- 평균 점수 4.0 이상 AND 모든 항목 3점 이상인 경우에만 "pass": true로 설정하세요.
- 마지막에 "이 이야기는~", "이처럼~" 같은 교훈 요약이 있으면 5번 항목에서 감점하세요.

[평가할 동화]
{story}

[요주의 문장] (1차 세이프가드에서 탐지됨)
{flagged}

[출력 형식] JSON만 출력하세요.
{{
  "scores": {{
    "narrative_context": <정수>,
    "role_modeling": <get_integer>,
    "moral_message": <정수>,
    "bias_stereotypes": <정수>,
    "logical_consistency": <정수>,
    "emotional_safety": <정수>,
    "setting_consistency": <정수>,
    "format_rules": <정수>
  }},
  "average": <평균 점수>,
  "pass": <true/false>,
  "feedback": {{
    "narrative_context": "...",
    "role_modeling": "...",
    "moral_message": "...",
    "bias_stereotypes": "...",
    "logical_consistency": "..."
  }},
  "revision_instruction": "<FAIL일 경우 Solar가 어떻게 고쳐야 할지 아주 구체적인 지침. PASS면 '없음'>"
}}"""


class StoryEvaluator:
    def __init__(self):
        api_key = os.getenv("UPSTAGE_API_KEY")
        if not api_key:
            raise ValueError(".env 파일에 UPSTAGE_API_KEY가 없습니다.")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.upstage.ai/v1"
        )
        print("Solar Pro 3 API 연결 완료!")

    def get_structured_guide(self, raw_input: str) -> str:
            """
            사용자의 모호한 입력을 분석하여 논문 기반의 구조화된 동화 설계도를 생성합니다. (0단계)
            """
            prompt = """사용자 입력을 바탕으로 동화의 '최소 뼈대'만 작성하세요. 
    상세한 대사나 줄거리를 미리 쓰지 마세요. 생성 모델이 창의적으로 쓸 여백을 남겨야 합니다.
    
    1. 인물: 이름, 나이(사용자가 성별을 입력했다면 삭제하세요)
    2. 소재: 갈등의 원인이 되는 물건 (예: 파란색 블록)
    3. 감정 키워드: (예: 서운함, 미안함)
    4. 교훈 키워드: (예: 양보)
    """
            
            response = self.client.chat.completions.create(
                model="solar-pro", # 구조화는 일반 모드로도 충분합니다
                messages=[{"role": "user", "content": prompt + raw_input}]
            )
            return response.choices[0].message.content

    def evaluate(self, story: str, flagged_sentences: list) -> dict:
        """
        동화를 CSM 기반 루브릭으로 2차 평가합니다.
        
        Args:
            story: 평가할 동화 전체 텍스트
            flagged_sentences: 1차에서 탐지된 요주의 문장 리스트
            
        Returns:
            평가 결과 딕셔너리
        """
        # 요주의 문장 정리
        if flagged_sentences:
            flagged_text = "\n".join([
                f"- [{s['category']}] {s['sentence']}"
                for s in flagged_sentences
            ])
        else:
            flagged_text = "없음"

        prompt = EVALUATION_PROMPT.format(
            story=story,
            flagged=flagged_text
        )

        # Solar Pro 3 API 호출
        stream = self.client.chat.completions.create(
            model="solar-pro3",
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="high",
            stream=True,
        )

        # 스트리밍 응답 수집
        response_text = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response_text += chunk.choices[0].delta.content

        # JSON 파싱
        import json
        import re
        # ```json ... ``` 형식 제거
        clean = re.sub(r"```json|```", "", response_text).strip()
        result = json.loads(clean)
        return result

    def revise(self, story: str, revision_instruction: str) -> str:
        """
        평가 결과를 바탕으로 동화를 직접 첨삭합니다.
        
        Args:
            story: 원본 동화 텍스트
            revision_instruction: 구체적인 수정 지시
            
        Returns:
            첨삭된 동화 텍스트
        """
        prompt = f"""당신은 6~7세 아동 동화 전문 베스트셀러 작가입니다. 
전달받은 초안은 현재 **이벤트 체인(Event Chain)의 논리가 깨져 있거나 캐릭터 속성이 충돌**하는 심각한 문제가 있습니다.

[수정 지침: 논문 기반 방법론]
1. 등장인물의 고정 속성(이름, 출신, 성격)을 먼저 확정하세요.
2. 사건의 인과관계(동사 체인)를 다시 설계하세요. 
   - 예: (원인) 친구가 장난감을 뺏음 -> (감정) 화가 남 -> (행동) 밀침 -> (반성) 미안해짐 -> (해결) 사과함
   - 절대 '도와주려고 했다'가 바로 '때렸다'로 이어지는 논리적 비약을 만들지 마세요.
3. 주인공에게 'High Agency(주체적 동사)'를 부여하여 스스로 갈등을 해결하게 하세요.

[수정 지시]
{revision_instruction}

[원본 초안]
{story}

[편집 원칙 - 반드시 따를 것]
0. 연극 대본이 아닌 줄글로 된 동화 형식을 따르세요. 6-7세에게 알맞은 300-350어절 이여야 합니다.
1. **부적절한 묘사 삭제**: '물고기 알을 빼앗아 삼킨다'는 등의 잔인하거나 엽기적인 묘사는 절대 금지입니다. 욕심을 부리는 행동을 '혼자 당근을 다 먹으려 한다' 정도로 순화하세요.
2. **비논리적 장치 제거**: 갑자기 등장하는 '협동 사다리', '협력 장치', '쪽지' 같은 작위적인 설정을 삭제하세요. 대신 친구와 대화하고 화해하는 자연스러운 과정을 넣으세요.
3. **일관성 유지**: 이야기가 중간에 리셋되거나 갑자기 캐릭터가 사라졌다 나타나는 비약을 없애고, 기-승-전-결이 매끄럽게 이어지게 하세요.
4. **결말 처리**: "이 이야기는~" 같은 설명은 절대 하지 말고, 등장인물이 함께 웃는 장면으로 끝내세요.
5. 맥락에 맞지 않는 단어가 나오지 않도록 하세요.
6. 반드시 동화의 형식을 갖추고 있도록 하세요.
7. 이야기가 동화로 적합한지 확인하세요.
8. 논리적 비약이 없는지 확인하세요.
9. 매우 엄격하게 평가하세요.
10. 사용하는 단어, 문장과 동화의 길이가 6-7세 아동에게 적합하도록 하세요. 

위 지시사항을 반영하여, 처음부터 끝까지 **사건의 체인이 논리적으로 완벽하게 이어지는** 동화로 다시 작성해 주세요. 
(설명 없이 동화 내용만 출력하세요.)

[수정된 동화]:"""

        stream = self.client.chat.completions.create(
            model="solar-pro3",
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="high",
            stream=True,
        )

        revised = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                revised += chunk.choices[0].delta.content

        return revised.strip()


# 단독 테스트용
if __name__ == "__main__":
    evaluator = StoryEvaluator()

    test_story = """
    토끼와 곰은 숲속 친구였어요.
    어느 날 곰이 토끼의 당근을 몰래 훔쳐 먹었어요.
    토끼는 매우 슬펐지만 곰에게 화를 내지 않고 왜 그랬는지 물어보았어요.
    곰은 배가 너무 고팠다고 사실대로 말했어요.
    토끼는 곰을 용서하고 함께 당근을 나눠 먹었어요.
    그 후로 둘은 더 좋은 친구가 되었답니다.
    """

    print("=" * 50)
    print("2차 평가 결과:")
    print("=" * 50)
    result = evaluator.evaluate(test_story, [])
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))