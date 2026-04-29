from src.generator import FairyTaleGenerator
from src.safeguard import SafeguardEvaluator
from src.evaluator import StoryEvaluator

MAX_LOOP = 3  # 최대 반복 횟수

class FairyTalePipeline:
    def __init__(self):
        print("\n[파이프라인 초기화 중...]\n")
        self.generator = FairyTaleGenerator()
        self.safeguard = SafeguardEvaluator()
        self.evaluator = StoryEvaluator()
        print("\n[파이프라인 초기화 완료!]\n")

    def run(self, topic: str) -> dict:
        """
        전체 파이프라인을 실행합니다.
        
        Args:
            topic: 동화 주제
            
        Returns:
            {
                "topic": 주제,
                "final_story": 최종 동화,
                "loop_count": 반복 횟수,
                "passed": 통과 여부,
                "final_scores": 최종 점수
            }
        """
        print(f"\n{'='*50}")
        print(f"주제: {topic}")
        print(f"{'='*50}\n")

        # ① 동화 초안 생성
        print("[1단계] 카나나 nano로 동화 생성 중...")
        story = self.generator.generate(topic)
        print(f"\n생성된 초안:\n{story}\n")

        for loop in range(1, MAX_LOOP + 1):
            print(f"\n{'─'*50}")
            print(f"[평가 루프 {loop}회차]")
            print(f"{'─'*50}")

            # ② 1차 평가 — 세이프가드
            print("\n[2단계] 카나나 세이프가드 1차 평가 중...")
            sg_result = self.safeguard.evaluate_story(story)
            flagged = sg_result["flagged_sentences"]
            print(f"요주의 문장: {len(flagged)}개")

            # ③ 2차 평가 — Solar API
            print("\n[3단계] Solar Pro 3 2차 평가 중...")
            eval_result = self.evaluator.evaluate(story, flagged)

            scores = eval_result["scores"]
            avg = eval_result["average"]
            passed = eval_result["pass"]

            print(f"\n📊 평가 점수:")
            print(f"  서사적 맥락:    {scores['narrative_context']}점")
            print(f"  아동 모델링:    {scores['role_modeling']}점")
            print(f"  도덕 메시지:    {scores['moral_message']}점")
            print(f"  편견·고정관념:  {scores['bias_stereotypes']}점")
            print(f"  평균:           {avg}점")
            print(f"  결과:           {'✅ PASS' if passed else '❌ FAIL'}")

            # ④ PASS → 완료
            if passed:
                print(f"\n🎉 {loop}회차에 통과!")
                return {
                    "topic": topic,
                    "final_story": story,
                    "loop_count": loop,
                    "passed": True,
                    "final_scores": eval_result,
                }

            # FAIL → Solar가 직접 첨삭
            if loop < MAX_LOOP:
                print(f"\n[4단계] Solar Pro 3 첨삭 중...")
                instruction = eval_result.get("revision_instruction", "전반적인 품질을 개선하세요.")
                print(f"수정 지시: {instruction}")
                story = self.evaluator.revise(story, instruction)
                print(f"\n첨삭된 동화:\n{story}\n")
            else:
                # 최대 반복 도달
                print(f"\n⚠️ 최대 반복 횟수({MAX_LOOP}회) 도달. 마지막 결과 반환.")
                return {
                    "topic": topic,
                    "final_story": story,
                    "loop_count": loop,
                    "passed": False,
                    "final_scores": eval_result,
                }