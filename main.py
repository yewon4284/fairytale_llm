"""
main.py
LLM 기반 아동 동화 생성 시스템의 진입점.

실행 방법:
    python main.py
    python main.py --request "친구를 때리면 안된다는 교훈을 주는 동화를 써줘."
    python main.py --generator nano          # 경량 모델로 빠른 실험
    python main.py --no-safeguard            # 세이프가드 건너뜀 (디버그용)
    python main.py --no-generator            # 카나나 로딩 건너뜀 (평가 단독 테스트)

[파이프라인 역할]
    기획     : Solar Pro       (UPSTAGE_API_KEY 필요)
    생성     : 카나나 1.5 8B  (기본) / 카나나 nano (--generator nano)
    1차 평가 : 카나나 세이프가드 8B
    2차 평가 : Solar Pro
"""

import argparse
import logging
import os
import sys

# Apple Silicon: MPS 통합 메모리 상한선 제거
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

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


# ── 인수 파싱 ─────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM 기반 아동 동화 자동 생성 및 안전성 평가 시스템"
    )
    parser.add_argument(
        "--request", type=str, default=None,
        help="동화 생성 요청 (미입력 시 대화형 입력)",
    )
    parser.add_argument(
        "--generator", type=str, default="8b",
        choices=["nano", "8b"],
        help="Generator 모델 선택 (기본: 8b / 경량: nano)",
    )
    parser.add_argument(
        "--no-safeguard", action="store_true",
        help="세이프가드 1차 평가 건너뜀 (디버그용)",
    )
    parser.add_argument(
        "--no-generator", action="store_true",
        help="카나나 로딩 건너뜀, 데이터셋 동화를 더미로 사용 (평가 단독 테스트)",
    )
    parser.add_argument(
        "--no-images", action="store_true",
        help="Stable Diffusion 이미지 생성 건너뜀",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="생성된 이미지 저장 디렉터리 (기본: outputs/)",
    )
    return parser.parse_args()


# ── 사용자 요청 입력 ──────────────────────────────────────────────────────────
def get_user_request(args: argparse.Namespace) -> str:
    if args.request:
        return args.request.strip()

    print("\n" + "=" * 70)
    print("  🌟 LLM 기반 아동 동화 생성 시스템")
    print("=" * 70)
    print("\n어떤 상황의 교훈을 담은 동화를 원하시나요?")
    print("예시:")
    print("  '우리 아이가 나비를 찢어죽였어. 그러면 안된다는 교훈을 주는 동화를 써줘.'")
    print("  '친구에게 욕설을 하면 안된다는 교훈을 주는 동화를 써줘.'")
    print("  '음식을 남기지 않아야 한다는 교훈을 담은 동화를 써줘.'")
    print()
    print("⚠  편향 단어 주의: 공주/왕자/아들/딸/특정 성별·인종 단어는 입력하지 마세요.")
    print("-" * 70)

    while True:
        request = input("요청 입력: ").strip()
        if request:
            return request
        print("요청을 입력해 주세요.")


# ── 퓨샷 데이터 로드 ──────────────────────────────────────────────────────────
def load_few_shot(n: int = 2) -> str:
    """
    데이터셋에서 동화를 n편 불러와 퓨샷 텍스트로 반환한다.
    데이터가 없으면 빈 문자열 반환 (퓨샷 없이 진행).
    """
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


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # API 키 확인
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        logger.error("UPSTAGE_API_KEY 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    # 사용자 요청 입력
    user_request = get_user_request(args)

    # 편향 단어 사전 검사
    from src.generator import check_bias
    biased = check_bias(user_request)
    if biased:
        print(f"\n❌ 편향 단어 감지: '{biased}'")
        print("   성별·인종 단어 없이 상황만 설명해 주세요.")
        sys.exit(0)

    print("\n⏳ 모델을 로딩합니다. 잠시 기다려 주세요...\n")

    # 퓨샷 데이터 로드
    few_shot_text = load_few_shot(n=2)
    if few_shot_text:
        logger.info("퓨샷 동화 2편 로드 완료")
    else:
        logger.info("퓨샷 없이 진행합니다")

    # ── Generator (카나나) ────────────────────────────────────────────────────
    from src.generator import KANANA_NANO, KANANA_15_8B, FairyTaleGenerator

    selected_model = KANANA_NANO if args.generator == "nano" else KANANA_15_8B

    if args.no_generator:
        from unittest.mock import MagicMock
        generator = MagicMock()
        generator.model_id = selected_model
        dummy_story = (
            few_shot_text.split("[참고 동화 1]\n")[-1].split("\n\n[참고 동화")[0].strip()
            if few_shot_text else "데이터셋에서 동화를 불러오지 못했습니다."
        )
        generator.generate.return_value = dummy_story
        logger.info("--no-generator 모드: 카나나 로딩 건너뜀, 더미 동화 사용")
    else:
        generator = FairyTaleGenerator(model_id=selected_model)

    # ── Safeguard (카나나 세이프가드 8B) ──────────────────────────────────────
    if args.no_safeguard:
        from unittest.mock import MagicMock
        safeguard = MagicMock()
        safeguard.evaluate_story.return_value = ([], [])
        logger.info("--no-safeguard 모드: 1차 평가 건너뜀")
    else:
        from src.safeguard import KananaSafeguard
        safeguard = KananaSafeguard()

    # ── Evaluator (Solar Pro) ─────────────────────────────────────────────────
    from src.evaluator import SolarEvaluator
    evaluator = SolarEvaluator(api_key=api_key)

    # ── 파이프라인 실행 ────────────────────────────────────────────────────────
    from src.pipeline import FairyTalePipeline, print_final_result

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