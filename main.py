from src.pipeline import FairyTalePipeline
import json

if __name__ == "__main__":
    pipeline = FairyTalePipeline()

    topic = """우리 아들이 유치원에서 장난감을 서로 가지고 놀겠다고 친구랑 싸웠대. 이 상황에 도움될만한 동화를 써줘"""
    result = pipeline.run(topic)

    print(f"\n{'='*50}")
    print("최종 결과")
    print(f"{'='*50}")
    print(f"주제: {result['topic']}")
    print(f"반복 횟수: {result['loop_count']}회")
    print(f"통과 여부: {'✅ PASS' if result['passed'] else '❌ FAIL'}")
    print(f"\n최종 동화:\n{result['final_story']}")
    print(f"\n최종 점수:\n{json.dumps(result['final_scores'], ensure_ascii=False, indent=2)}")