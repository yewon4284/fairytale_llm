"""
test.py
Solar Pro 단독 동화 생성 시스템 진입점 (비교 실험용).

실행 방법:
    python test.py
    python test.py --request "친구를 때리면 안된다는 교훈을 주는 동화를 써줘."
    python test.py --no-images
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solar Pro 단독 동화 생성 시스템 (비교 실험용)"
    )
    parser.add_argument(
        "--request", type=str, default=None,
        help="동화 생성 요청 (미입력 시 대화형 입력)",
    )
    parser.add_argument(
        "--no-images", action="store_true",
        help="Pollinations.ai 이미지 생성 건너뜀",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs_solar",
        help="생성된 이미지 저장 디렉터리 (기본: outputs_solar/)",
    )
    return parser.parse_args()


def get_user_request(args: argparse.Namespace) -> str:
    if args.request:
        return args.request.strip()

    print("\n" + "=" * 70)
    print("  🌟 Solar Pro 단독 동화 생성 시스템 (비교 실험용)")
    print("=" * 70)
    print("\n어떤 상황의 교훈을 담은 동화를 원하시나요?")
    print("예시:")
    print("  '우리 아이가 나비를 찢어죽였어. 그러면 안된다는 교훈을 주는 동화를 써줘.'")
    print("  '친구에게 욕설을 하면 안된다는 교훈을 주는 동화를 써줘.'")
    print()
    print("⚠  편향 단어 주의: 공주/왕자/아들/딸/특정 성별·인종 단어는 입력하지 마세요.")
    print("-" * 70)

    while True:
        request = input("요청 입력: ").strip()
        if request:
            return request
        print("요청을 입력해 주세요.")


def load_few_shot(n: int = 2) -> str:
    try:
        from src.data_loader import load_communication_tales, get_sample_texts
        tales   = load_communication_tales()
        samples = get_sample_texts(tales, n=n)
        if not samples:
            return ""
        lines = [f"[참고 동화 {i}]\n{text}" for i, text in enumerate(samples, 1)]
        return "\n\n".join(lines)
    except Exception as e:
        logger.warning(f"퓨샷 데이터 로드 실패 — 퓨샷 없이 진행합니다. ({e})")
        return ""


def main():
    args = parse_args()

    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        logger.error("UPSTAGE_API_KEY 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    user_request = get_user_request(args)

    from src.generator import check_bias
    biased = check_bias(user_request)
    if biased:
        print(f"\n❌ 편향 단어 감지: '{biased}'")
        print("   성별·인종 단어 없이 상황만 설명해 주세요.")
        sys.exit(0)

    few_shot_text = load_few_shot(n=2)
    if few_shot_text:
        logger.info("퓨샷 동화 2편 로드 완료")
    else:
        logger.info("퓨샷 없이 진행합니다")

    from test_src.solar_generator import SolarGenerator
    from src.evaluator import SolarEvaluator
    from test_src.solar_pipeline import SolarOnlyPipeline, print_final_result

    generator = SolarGenerator(api_key=api_key)
    evaluator = SolarEvaluator(api_key=api_key)

    pipeline = SolarOnlyPipeline(
        generator=generator,
        evaluator=evaluator,
        few_shot_text=few_shot_text,
    )

    try:
        result = pipeline.run(user_request)
        print_final_result(result)
    except ValueError as e:
        print(f"\n❌ 오류: {e}")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n사용자가 중단했습니다.")
        sys.exit(0)

    # ── 이미지 생성 (Pollinations.ai) ────────────────────────────────────────
    if not args.no_images:
        print("\n" + "=" * 70)
        print("  🎨 동화 삽화 생성 (Pollinations.ai)")
        print("=" * 70)
        try:
            from src.image_generator import FairyTaleImageGenerator
            img_gen = FairyTaleImageGenerator(
                api_key=api_key,
                output_dir=args.output_dir,
            )
            paths, scenes = img_gen.generate(
                story=result.final_story,
                plan=result.final_plan,
            )
            print(f"\n✅ 이미지 {len(paths)}장 생성 완료:")
            for path, scene in zip(paths, scenes):
                print(f"  • {scene.get('scene_ko', '')} → {path}")
        except Exception as e:
            logger.error(f"이미지 생성 중 오류: {e}")
            print(f"\n⚠ 이미지 생성 중 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()
