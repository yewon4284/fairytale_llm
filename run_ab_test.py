"""
run_ab_test.py
퓨샷(few-shot) on/off 비교 실험 러너.

test_topics.json에 있는 각 주제에 대해
  1) 퓨샷 있음 (few_shot_on)
  2) 퓨샷 없음 (few_shot_off)
두 조건으로 파이프라인을 돌려 결과를 ab_results.json에 누적 저장한다.

모델(생성기·세이프가드·Solar)은 한 번만 로딩해서 두 조건 모두에 재사용한다.
중간에 중단돼도 이미 끝난 (topic_id, condition)은 다시 안 돌리고 이어서 진행한다.

사용법:
    python run_ab_test.py                          # 25개 전부, 두 조건
    python run_ab_test.py --limit 3                # 앞 3개 주제만 (동작 확인용)
    python run_ab_test.py --no-safeguard            # 세이프가드 생략 (빠른 디버그용)
    python run_ab_test.py --summary                 # 실행 없이 기존 결과만 집계해서 출력
"""

import argparse
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

SCORE_KEYS = ["서사적_맥락", "아동_모델링", "도덕_메시지", "편견_고정관념", "언어_표현", "교육적_가치"]
CONDITIONS = ["few_shot_on", "few_shot_off"]


def load_topics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_results(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_results(path, results):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def summarize_pipeline_result(topic_id, condition, request, result, elapsed):
    """PipelineResult -> 저장용 dict (best_attempt 기준)"""
    rec = next((r for r in result.attempts if r.attempt == result.best_attempt), None)
    if rec is None and result.attempts:
        rec = result.attempts[-1]
    eval_result = rec.eval_result if rec else {}
    scores = eval_result.get("scores", {})
    return {
        "topic_id": topic_id,
        "condition": condition,
        "request": request,
        "passed": result.passed,
        "total_attempts": result.total_attempts,
        "best_attempt": result.best_attempt,
        "average_score": eval_result.get("average_score", 0),
        "min_score": eval_result.get("min_score", 0),
        "body_safety_pass": eval_result.get("body_safety_pass", True),
        "scores": {k: scores.get(k) for k in SCORE_KEYS},
        "flagged_count": len(rec.flagged_sentences) if rec else 0,
        "char_count": len(result.final_story.replace(" ", "")),
        "elapsed_sec": round(elapsed, 1),
        "final_plan": result.final_plan,
        "final_story": result.final_story,
    }


def print_summary(results):
    print("\n" + "=" * 70)
    print("  퓨샷 on/off 비교 결과")
    print("=" * 70)
    for cond in CONDITIONS:
        rows = [r for r in results if r["condition"] == cond]
        if not rows:
            print(f"\n[{cond}] 결과 없음")
            continue
        n = len(rows)
        pass_rate = sum(r["passed"] for r in rows) / n
        avg_score = sum(r["average_score"] for r in rows) / n
        avg_attempts = sum(r["total_attempts"] for r in rows) / n
        avg_flagged = sum(r["flagged_count"] for r in rows) / n
        avg_chars = sum(r["char_count"] for r in rows) / n

        print(f"\n[{cond}]  (n={n})")
        print(f"  합격률(pass rate):     {pass_rate*100:.1f}%")
        print(f"  평균 점수:             {avg_score:.2f} / 5.00")
        print(f"  평균 재시도 횟수:      {avg_attempts:.2f}회")
        print(f"  평균 세이프가드 태깅:  {avg_flagged:.2f}개 문장")
        print(f"  평균 글자수(공백제외): {avg_chars:.0f}자")
        print("  항목별 평균 점수:")
        for k in SCORE_KEYS:
            vals = [r["scores"].get(k) for r in rows if r["scores"].get(k) is not None]
            if vals:
                print(f"    - {k}: {sum(vals)/len(vals):.2f}")
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", default="test_topics.json")
    parser.add_argument("--output", default="ab_results.json")
    parser.add_argument("--limit", type=int, default=None, help="앞에서 N개 주제만 테스트")
    parser.add_argument("--generator", default="1.5-8b", choices=["nano", "1.5-8b"])
    parser.add_argument("--no-safeguard", action="store_true")
    parser.add_argument("--few-shot-n", type=int, default=2)
    parser.add_argument("--summary", action="store_true", help="실행 없이 기존 결과만 집계")
    args = parser.parse_args()

    if args.summary:
        results = load_existing_results(args.output)
        if not results:
            print(f"{args.output}에 결과가 없습니다.")
            return
        print_summary(results)
        return

    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        logger.error("UPSTAGE_API_KEY 환경변수가 없습니다 (.env 확인).")
        sys.exit(1)

    topics = load_topics(args.topics)
    if args.limit:
        topics = topics[: args.limit]

    from src.generator import KANANA_NANO, KANANA_15_8B, FairyTaleGenerator
    from src.pipeline import FairyTalePipeline
    from src.evaluator import SolarEvaluator
    from src.data_loader import get_reference_stories

    selected_model = KANANA_15_8B if args.generator == "1.5-8b" else KANANA_NANO
    logger.info(f"Generator 로딩: {selected_model}")
    generator = FairyTaleGenerator(model_id=selected_model)

    if args.no_safeguard:
        from unittest.mock import MagicMock
        safeguard = MagicMock()
        safeguard.evaluate_story.return_value = ([], [])
        logger.info("세이프가드 생략 모드")
    else:
        from src.safeguard import KananaSafeguard
        logger.info("Safeguard 로딩...")
        safeguard = KananaSafeguard()

    evaluator = SolarEvaluator(api_key=api_key)

    # 퓨샷 텍스트 준비 (data_sorted 1~20번 고정, seed=42로 n편 무작위 — README 기준과 동일)
    refs = get_reference_stories(n=args.few_shot_n, classification="의사소통")
    lines = []
    for i, ref in enumerate(refs, 1):
        title = ref.get("title", "")
        text = ref.get("text", "").strip()
        if text:
            header = f"[참고 동화 {i}] {title}" if title else f"[참고 동화 {i}]"
            lines.append(f"{header}\n{text}")
    few_shot_text = "\n\n".join(lines)
    logger.info(f"퓨샷 {len(refs)}편 로드 완료 ({len(few_shot_text)}자)")
    if not few_shot_text:
        logger.warning("퓨샷 텍스트가 비어있습니다 — data_sorted 폴더를 확인하세요.")

    pipelines = {
        "few_shot_on": FairyTalePipeline(generator, safeguard, evaluator, few_shot_text=few_shot_text),
        "few_shot_off": FairyTalePipeline(generator, safeguard, evaluator, few_shot_text=""),
    }

    results = load_existing_results(args.output)
    done = {(r["topic_id"], r["condition"]) for r in results}

    total_runs = len(topics) * len(CONDITIONS)
    run_idx = len(done)

    for topic in topics:
        tid = topic["id"]
        request = topic["topic"]

        for condition in CONDITIONS:
            if (tid, condition) in done:
                continue
            run_idx += 1
            logger.info(f"[{run_idx}/{total_runs}] topic {tid} ({condition}) 시작 — {request}")

            start = time.time()
            try:
                result = pipelines[condition].run(request)
            except ValueError as e:
                logger.warning(f"topic {tid} ({condition}) 편향 단어 감지, 스킵: {e}")
                continue
            except Exception as e:
                logger.exception(f"topic {tid} ({condition}) 실패, 스킵: {e}")
                continue
            elapsed = time.time() - start

            record = summarize_pipeline_result(tid, condition, request, result, elapsed)
            results.append(record)
            save_results(args.output, results)

            logger.info(
                f"  -> {'PASS' if record['passed'] else 'FAIL'} "
                f"평균 {record['average_score']} / 시도 {record['total_attempts']}회 / {elapsed:.0f}초"
            )

    logger.info(f"전체 완료 -> {args.output}")
    print_summary(results)


if __name__ == "__main__":
    main()
