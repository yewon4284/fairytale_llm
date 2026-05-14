"""
main.py
LLM 기반 아동 동화 자동 생성 및 안전성 평가 시스템

사용법:
    python main.py
    python main.py --request "우리 아이가 거짓말을 했어. 정직함의 교훈을 담은 동화를 써줘."
    python main.py --output result.json
"""

import argparse
import json
import os
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="동화 생성 파이프라인")
    parser.add_argument(
        "--request", type=str,
        default="우리 아이가 나비를 찢어죽였어. 그러면 안된다는 교훈을 느낄 수 있는 동화를 써줘.",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=== 전체 파이프라인 실행 ===\n")

    from src.generator import FairyTaleGenerator
    from src.safeguard import KananaSafeguard
    from src.evaluator import SolarEvaluator
    from src.pipeline import FairyTalePipeline

    generator = FairyTaleGenerator()
    safeguard = KananaSafeguard()
    evaluator = SolarEvaluator()
    pipeline = FairyTalePipeline(
        generator=generator,
        safeguard=safeguard,
        evaluator=evaluator,
        data_dir="data",
    )

    result = pipeline.run(args.request)

    if args.output:
        _save_result(result, args.output)


def _save_result(result, path: str):
    data = {
        "final_tale": result.final_tale,
        "passed": result.passed,
        "total_time_sec": result.total_time_sec,
        "evaluation_history": [
            {
                "scores": e.scores,
                "average": e.average,
                "passed": e.passed,
                "overall_feedback": e.overall_feedback,
                "rewrite_instruction": e.rewrite_instruction,
            }
            for e in result.evaluation_history
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {path}")


if __name__ == "__main__":
    main()