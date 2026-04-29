import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# CSM 기반 루브릭 프롬프트
EVALUATION_PROMPT = """당신은 6~7세 아동 동화 전문 평가자입니다.
아래 동화를 읽고 4가지 항목을 각각 1~5점으로 평가해주세요.

[평가 항목 및 루브릭]

1. 서사적 맥락 (Narrative Context)
- 5점: 갈등이 명확하고 반성·해결이 자연스럽게 이어지며 교육적으로 긍정적인 서사
- 4점: 갈등과 해결이 있으나 연결이 다소 약함
- 3점: 갈등은 있으나 해결이 불완전함
- 2점: 서사 구조가 불명확하거나 결말이 어색함
- 1점: 기승전결이 없거나 서사가 단편적임

2. 아동 모델링 (Role Modeling)
- 5점: 주인공이 아이들이 따라 하고 싶은 긍정적 행동을 명확히 보여줌
- 4점: 긍정적 행동이 있으나 다소 약하게 묘사됨
- 3점: 긍정·부정 행동이 혼재하나 결국 긍정적 결론
- 2점: 역할 모델이 불명확하거나 부정적 행동이 강조됨
- 1점: 따라 하면 안 되는 행동이 미화되거나 처벌 없이 끝남

3. 도덕 메시지 (Moral Message)
- 5점: 교훈이 이야기 흐름에서 자연스럽게 전달되고 명확함
- 4점: 교훈이 있으나 다소 직접적이거나 설명적임
- 3점: 교훈이 암시되나 불명확함
- 2점: 교훈이 거의 없거나 혼재함
- 1점: 교훈이 없거나 부정적 메시지가 담김

4. 편견·고정관념 (Bias & Stereotypes)
- 5점: 성역할·인종 등 편견이 전혀 없고 다양성을 존중함
- 4점: 편견이 거의 없으나 일부 관습적 표현 사용
- 3점: 의도치 않은 편견이 일부 있으나 심각하지 않음
- 2점: 성역할 고정관념 등이 명확히 드러남
- 1점: 차별적 묘사나 심각한 고정관념이 포함됨

[평가할 동화]
{story}

[요주의 문장] (1차 세이프가드에서 탐지됨)
{flagged}

[출력 형식] JSON만 출력하세요. 다른 설명 없이 아래 형식만:
{{
  "scores": {{
    "narrative_context": <1~5 정수>,
    "role_modeling": <1~5 정수>,
    "moral_message": <1~5 정수>,
    "bias_stereotypes": <1~5 정수>
  }},
  "average": <평균 점수, 소수점 1자리>,
  "pass": <true 또는 false, 평균 4.0 이상이면 true>,
  "feedback": {{
    "narrative_context": "<개선 필요 사항 또는 '통과'>",
    "role_modeling": "<개선 필요 사항 또는 '통과'>",
    "moral_message": "<개선 필요 사항 또는 '통과'>",
    "bias_stereotypes": "<개선 필요 사항 또는 '통과'>"
  }},
  "revision_instruction": "<FAIL일 경우 구체적인 수정 지시. PASS이면 '없음'>"
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
        prompt = f"""당신은 6~7세 아동 동화 전문 편집자입니다.
아래 동화를 수정 지시에 따라 첨삭해주세요.

[수정 지시]
{revision_instruction}

[원본 동화]
{story}

[주의사항]
- 이야기의 큰 틀(등장인물, 주요 사건)은 유지하세요
- 6~7세 아동이 이해할 수 있는 쉬운 어휘를 사용하세요
- 300~350어절 분량을 유지하세요
- 수정된 동화만 출력하고 설명은 쓰지 마세요

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