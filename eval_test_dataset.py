"""
eval_test_dataset.py
카나나(생성) 단계를 건너뛰고, all_data의 기존 동화에 대해
평가 파이프라인(1차: kanana-safeguard-8b 세이프가드, 2차: Solar Pro3 맥락 평가)만 실행하여
동화별 SAFE/UNSAFE(=PASS/FAIL) 결과를 산출한다.

테스트셋 = all_data 전체(441편) - data_sorted에 있는 퓨샷 20편 (파일명 기준 제외) = 421편

사용법 (fairytale_llm 저장소 루트에서 실행):
    python eval_test_dataset.py                     # 421편 전체 실행 (중단 후 재실행 시 이어서 진행)
    python eval_test_dataset.py --limit 5            # 앞 5편만 (동작 확인용)
    python eval_test_dataset.py --no-safeguard        # 세이프가드 생략, Solar 2차 평가만
    python eval_test_dataset.py --output my_result.json
"""

import argparse
import glob
import json
import logging
import os
import sys
import time

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

load_dotenv()

ALL_DATA_DIR = "all_data"      # 전체 441편
FEWSHOT_DIR = "data_sorted"    # 퓨샷 고정 20편 (제외 대상)


def get_test_filepaths(include_fewshot=False):
    """all_data 전체 경로 목록.
    include_fewshot=False(기본)면 기존처럼 data_sorted(퓨샷 20편)와 파일명이 겹치는
    항목을 제외해 421편만 반환 (퓨샷 실험과 섞이지 않게). include_fewshot=True면
    제외 없이 441편 전부 반환 (퓨샷 실험을 더 이상 안 할 때 — 성능평가 벤치마크는
    421편이 아니라 441편 전체를 쓰기로 함)."""
    all_paths = sorted(glob.glob(os.path.join(ALL_DATA_DIR, "*.json")))
    if include_fewshot:
        logger.info(f"all_data {len(all_paths)}편 전부를 테스트셋으로 사용 (퓨샷 20편 포함)")
        return all_paths
    fewshot_names = {
        os.path.basename(p) for p in glob.glob(os.path.join(FEWSHOT_DIR, "*.json"))
    }
    test_paths = [p for p in all_paths if os.path.basename(p) not in fewshot_names]
    logger.info(
        f"all_data {len(all_paths)}편 중 퓨샷 {len(fewshot_names)}편 제외 -> 테스트셋 {len(test_paths)}편"
    )
    return test_paths


def load_test_stories(include_fewshot=False):
    from src.data_loader import story_to_text  # JSON -> 본문 텍스트 변환

    stories = []
    for p in get_test_filepaths(include_fewshot=include_fewshot):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            text = story_to_text(data)
            if not text:
                logger.warning(f"{p}: 본문(paragraphInfo) 없음 -> 건너뜀")
                continue
            stories.append({
                "filename": os.path.basename(p),
                "title": data.get("title", ""),
                "text": text,
            })
        except Exception as e:
            logger.warning(f"{p} 로드 실패: {e}")
    return stories


def load_existing(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save(path, results):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="eval_results.json")
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N편만 테스트 (동작 확인용)")
    ap.add_argument("--no-safeguard", action="store_true", help="1차 세이프가드 생략 (Solar만)")
    ap.add_argument("--include-fewshot", action="store_true",
                     help="퓨샷 20편도 포함해 all_data 441편 전체를 테스트셋으로 사용 "
                          "(기본은 421편 — 퓨샷 20편 제외)")
    ap.add_argument("--adapter-path", default=None,
                     help="S5_ADAPTER_PATH 대신 사용할 LoRA 어댑터 경로 (예: 재보정 어댑터를 "
                          "프로덕션 기본값으로 바꾸기 전에 테스트할 때). 미지정 시 기존 기본 동작.")
    ap.add_argument("--evaluator", choices=["solar", "hcx"], default="solar",
                     help="2차 평가 백엔드 선택 (solar=Upstage Solar, hcx=Naver HyperCLOVA X)")
    ap.add_argument("--solar-model", default=None,
                     help="--evaluator solar일 때 모델명 오버라이드 (예: solar-pro3, solar-pro4). "
                          "미지정 시 evaluator.py의 기본 SOLAR_MODEL 사용")
    ap.add_argument("--hcx-model", default="HCX-007",
                     help="--evaluator hcx일 때 모델명 (기본 HCX-007)")
    ap.add_argument("--hcx-api-key", default=None,
                     help="HCX API 키. 미지정 시 환경변수 HCX_API_KEY 사용")
    ap.add_argument("--eval-temperature", type=float, default=0.0,
                     help="2차 평가 API 호출 temperature. 모델 비교 실험에서는 표본 변동을 "
                          "줄이려고 기본값을 0으로 낮춤 (기존 0.3)")
    args = ap.parse_args()

    if args.evaluator == "hcx":
        hcx_key = args.hcx_api_key or os.getenv("HCX_API_KEY")
        if not hcx_key:
            logger.error("HCX_API_KEY 환경변수(또는 --hcx-api-key)가 없습니다 (.env 확인).")
            sys.exit(1)
    else:
        api_key = os.getenv("UPSTAGE_API_KEY")
        if not api_key:
            logger.error("UPSTAGE_API_KEY 환경변수가 없습니다 (.env 확인).")
            sys.exit(1)

    stories = load_test_stories(include_fewshot=args.include_fewshot)
    if args.limit:
        stories = stories[: args.limit]
    logger.info(f"평가 대상 {len(stories)}편")

    if args.no_safeguard:
        from unittest.mock import MagicMock
        safeguard = MagicMock()
        safeguard.evaluate_story.return_value = ([], [])
        logger.info("세이프가드 생략 모드")
    else:
        from src.safeguard import KananaSafeguard
        logger.info("Safeguard(kanana-safeguard-8b) 로딩..."
                     + (f" (어댑터 오버라이드: {args.adapter_path})" if args.adapter_path else ""))
        safeguard = KananaSafeguard(adapter_path=args.adapter_path)

    if args.evaluator == "hcx":
        from src.evaluator_hcx import NaverHCXEvaluator
        logger.info(f"2차 평가 백엔드: HyperCLOVA X ({args.hcx_model}), temperature={args.eval_temperature}")
        evaluator = NaverHCXEvaluator(
            api_key=hcx_key, model=args.hcx_model, eval_temperature=args.eval_temperature
        )
    else:
        from src.evaluator import SolarEvaluator
        model_label = args.solar_model or "기본값"
        logger.info(f"2차 평가 백엔드: Solar ({model_label}), temperature={args.eval_temperature}")
        evaluator = SolarEvaluator(
            api_key=api_key, model=args.solar_model, eval_temperature=args.eval_temperature
        )

    results = load_existing(args.output)
    done = {r["filename"] for r in results}

    for i, story in enumerate(stories, 1):
        if story["filename"] in done:
            continue
        logger.info(f"[{i}/{len(stories)}] {story['filename']} ({story['title']}) 평가 중...")
        start = time.time()
        try:
            sentences, flagged = safeguard.evaluate_story(story["text"])
            # 생성 파이프라인의 user_request 자리에는 원 데이터셋 제목을 맥락으로 전달
            eval_result, passed, summary = evaluator.evaluate(
                story["text"], flagged, user_request=story["title"]
            )
        except Exception as e:
            logger.exception(f"{story['filename']} 평가 실패, 건너뜀: {e}")
            continue
        elapsed = time.time() - start

        record = {
            "filename": story["filename"],
            "title": story["title"],
            "verdict": "SAFE" if passed else "UNSAFE",
            "average_score": eval_result.get("average_score", 0),
            "min_score": eval_result.get("min_score", 0),
            "body_safety_pass": eval_result.get("body_safety_pass", True),
            "body_safety_note": eval_result.get("body_safety_note", ""),
            "flagged_categories": sorted({f["category"] for f in flagged}),
            "fail_reasons": eval_result.get("fail_reasons", []),
            "elapsed_sec": round(elapsed, 1),
        }
        results.append(record)
        save(args.output, results)
        logger.info(f"  -> {record['verdict']} (평균 {record['average_score']}점) / {elapsed:.0f}초")

        # 장시간 반복 실행 시 CUDA 메모리 파편화 완화
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    n_safe = sum(1 for r in results if r["verdict"] == "SAFE")
    n_unsafe = len(results) - n_safe

    print("\n" + "=" * 70)
    print("  동화별 SAFE / UNSAFE 결과")
    print("=" * 70)
    for r in results:
        print(f"  [{r['verdict']:6s}] {r['filename']:35s} {r['title']:20s} 평균 {r['average_score']}점")
    print("-" * 70)
    print(f"  SAFE {n_safe}편 / UNSAFE {n_unsafe}편 / 전체 {len(results)}편")
    print(f"  상세 결과 저장: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
