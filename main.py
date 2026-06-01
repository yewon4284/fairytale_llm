"""
main.py
LLM 기반 아동 동화 생성 시스템의 진입점.

실행 방법:
    python main.py                     # 대화형 — 동화 생성 후 계속 여부 묻기
    python main.py --request "..."     # 요청 직접 전달 (1회 실행 후 종료)
    python main.py --generator nano    # 비교 실험용 (nano 모델)
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
load_dotenv()


def get_user_request(args) -> str:
    if args.request:
        return args.request.strip()

    print("\n" + "=" * 70)
    print("  LLM 기반 아동 동화 생성 시스템")
    print("=" * 70)
    print("\n어떤 상황의 교훈을 담은 동화를 원하시나요?")
    print("예시: '편식을 하면 좋지 않다는 교훈을 주는 동화를 써줘.'")
    print("      '친구에게 욕설을 하면 안된다는 교훈을 주는 동화를 써줘.'")
    print()
    print("⚠  편향 단어 주의: 공주/왕자/아들/딸/특정 성별·인종 단어는 입력하지 마세요.")
    print("-" * 70)

    while True:
        request = input("요청 입력: ").strip()
        if request:
            return request
        print("요청을 입력해 주세요.")


def load_few_shot(n: int = 2) -> str:
    """
    데이터셋에서 의사소통 분류 동화를 n편 불러와 퓨샷 텍스트로 반환한다.
    데이터가 없으면 빈 문자열 반환 (퓨샷 없이 진행).
    """
    try:
        from src.data_loader import get_reference_stories
        refs = get_reference_stories(n=n, classification="의사소통")
        if not refs:
            return ""
        lines = []
        for i, ref in enumerate(refs, 1):
            title = ref.get("title", "")
            text = ref.get("text", "").strip()
            if text:
                header = f"[참고 동화 {i}] {title}" if title else f"[참고 동화 {i}]"
                lines.append(f"{header}\n{text}")
        return "\n\n".join(lines)
    except Exception as e:
        logger.warning(f"퓨샷 데이터 로드 실패 (퓨샷 없이 진행): {e}")
        return ""


def run_once(args, api_key, generator, safeguard, evaluator, few_shot_text):
    """동화 한 편을 생성·평가하고 결과를 출력한다."""
    from src.generator import check_bias
    from src.pipeline import FairyTalePipeline, print_final_result

    user_request = get_user_request(args)

    # 편향 단어 검사
    biased = check_bias(user_request)
    if biased:
        print(f"\n❌ 편향 단어 감지: '{biased}'")
        print("   성별·인종 단어 없이 상황만 설명해 주세요.")
        return  # 종료하지 않고 다음 루프로

    pipeline = FairyTalePipeline(
        generator=generator,
        safeguard=safeguard,
        evaluator=evaluator,
        few_shot_text=few_shot_text,
    )

    try:
        result = pipeline.run(user_request)
        print_final_result(result)
    except ValueError as e:
        print(f"\n❌ 오류: {e}")
    except KeyboardInterrupt:
        raise  # 바깥 루프에서 처리


def main():
    parser = argparse.ArgumentParser(
        description="LLM 기반 아동 동화 자동 생성 및 안전성 평가 시스템"
    )
    parser.add_argument("--request", type=str, default=None,
                        help="동화 생성 요청 (입력 시 1회 실행 후 종료)")
    parser.add_argument("--generator", type=str, default="1.5-8b",
                        choices=["nano", "1.5-8b"],
                        help="Generator 모델 (기본: 1.5-8b / 비교 실험: nano)")
    parser.add_argument("--no-safeguard", action="store_true",
                        help="세이프가드 1차 평가 건너뜀 (디버그용)")
    parser.add_argument("--no-generator", action="store_true",
                        help="카나나 로딩 건너뜀, 데이터셋 동화를 더미로 사용 (디버그용)")
    args = parser.parse_args()

    # ── API 키 확인 ────────────────────────────────────────────────────────────
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        logger.error("UPSTAGE_API_KEY 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    # ── 퓨샷 데이터 로드 (모델 로딩 전, 1회만) ────────────────────────────────
    print("\n⏳ 모델을 로딩합니다. 잠시 기다려 주세요...\n")
    few_shot_text = load_few_shot(n=2)
    if few_shot_text:
        logger.info("퓨샷 동화 2편 로드 완료")
    else:
        logger.info("퓨샷 없이 진행합니다")

    # ── 모델 로딩 (1회만) ─────────────────────────────────────────────────────
    from src.generator import KANANA_NANO, KANANA_15_8B, FairyTaleGenerator
    selected_model = KANANA_15_8B if args.generator == "1.5-8b" else KANANA_NANO

    if args.no_generator:
        from unittest.mock import MagicMock
        generator = MagicMock()
        generator.model_id = selected_model
        dummy_story = few_shot_text.split("[참고 동화 1]\n")[-1].split("\n\n[참고 동화")[0].strip()
        generator.generate.return_value = dummy_story or "데이터셋에서 동화를 불러오지 못했습니다."
    else:
        generator = FairyTaleGenerator(model_id=selected_model)

    if args.no_safeguard:
        from unittest.mock import MagicMock
        safeguard = MagicMock()
        safeguard.evaluate_story.return_value = ([], [])
    else:
        from src.safeguard import KananaSafeguard
        safeguard = KananaSafeguard()

    from src.evaluator import SolarEvaluator
    evaluator = SolarEvaluator(api_key=api_key)

    # ── 실행 ──────────────────────────────────────────────────────────────────
    # --request 인자가 있으면 1회 실행 후 종료
    # 없으면 동화 생성 후 계속 여부를 묻고 반복
    if args.request:
        run_once(args, api_key, generator, safeguard, evaluator, few_shot_text)
    else:
        while True:
            try:
                run_once(args, api_key, generator, safeguard, evaluator, few_shot_text)
                print("\n" + "=" * 70)
                again = input("다른 동화를 만드시겠습니까? (y/n): ").strip().lower()
                if again != "y":
                    print("시스템을 종료합니다.")
                    break
            except KeyboardInterrupt:
                print("\n\n사용자가 중단했습니다.")
                break


if __name__ == "__main__":
    main()