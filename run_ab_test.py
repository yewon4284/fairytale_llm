"""
run_ab_test.py
퓨샷(few-shot) on/off — 출력 형식(글자수/문단수/문장수/단어수) 일관성 비교 러너.

목적: 퓨샷을 넣었을 때 생성되는 동화의 "형식적" 측면(길이, 문단 구성, 문장 길이 등)이
      넣지 않았을 때보다 얼마나 더 일정하게(변동폭이 작게) 나오는지 수치로 비교한다.
      점수(CSM 평가)는 참고용으로만 같이 기록한다.

test_topics.json에 있는 각 주제에 대해
  1) 퓨샷 있음 (few_shot_on)
  2) 퓨샷 없음 (few_shot_off)
두 조건으로 파이프라인을 돌려 결과를 ab_results.json에 누적 저장한다.

형식 지표는 "1차 생성본"(attempt 1, 세이프가드 재작성 루프를 타기 전 원본)을 기준으로 잰다.
재시도를 거치면 rewrite_hint가 길이에 영향을 줘서 순수한 "퓨샷 효과"가 아니게 되기 때문.

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
import re
import statistics
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

# safeguard.py의 문장 분리 정규식과 동일 기준 사용 (마침표/물음표/느낌표/'요' 뒤 공백)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?요])\s+")

# 형식 지표로 집계할 필드와 표시 이름
FORMAT_METRIC_LABELS = {
    "char_count": "글자수(공백제외)",
    "char_count_ws": "글자수(공백포함)",
    "word_count": "단어수(어절)",
    "sentence_count": "문장수",
    "paragraph_count": "문단수",
    "avg_sentence_len": "문장당 평균 글자수",
    "avg_paragraph_len": "문단당 평균 글자수",
}


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


def compute_format_metrics(story: str) -> dict:
    """동화 텍스트 하나에서 형식적 지표를 뽑아낸다."""
    story = story.strip()
    char_count = len(story.replace(" ", "").replace("\n", ""))
    char_count_ws = len(story.replace("\n", ""))
    word_count = len(story.split())

    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(story) if s.strip()]
    sentence_count = len(sentences) or 1

    paragraphs = [p.strip() for p in story.split("\n") if p.strip()]
    paragraph_count = len(paragraphs) or 1

    return {
        "char_count": char_count,
        "char_count_ws": char_count_ws,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "avg_sentence_len": round(char_count / sentence_count, 1),
        "avg_paragraph_len": round(char_count / paragraph_count, 1),
    }


def summarize_pipeline_result(topic_id, condition, request, result, elapsed):
    """PipelineResult -> 저장용 dict.
    format_first  : attempt 1(원본, 재작성 전) 기준 형식 지표 — 퓨샷 효과 비교용 핵심 데이터
    format_final  : 최종 채택본 기준 형식 지표 — 참고용
    """
    first_rec = next((r for r in result.attempts if r.attempt == 1), None)
    best_rec = next((r for r in result.attempts if r.attempt == result.best_attempt), None)
    if best_rec is None and result.attempts:
        best_rec = result.attempts[-1]

    first_story = first_rec.story if first_rec else result.final_story
    eval_result = best_rec.eval_result if best_rec else {}
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
        "scores": {k: scores.get(k) for k in SCORE_KEYS},
        "elapsed_sec": round(elapsed, 1),
        "format_first": compute_format_metrics(first_story),
        "format_final": compute_format_metrics(result.final_story),
        "first_story": first_story,
        "final_story": result.final_story,
    }


def _mean_std_cv(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    cv = (std / mean * 100) if mean else 0.0  # 변동계수(%) — 낮을수록 일정함
    return mean, std, cv


def print_summary(results):
    print("\n" + "=" * 78)
    print("  퓨샷 on/off — 형식(길이/문단/문장) 일관성 비교  (attempt 1, 원본 기준)")
    print("=" * 78)

    stats_by_cond = {}
    for cond in CONDITIONS:
        rows = [r for r in results if r["condition"] == cond]
        if not rows:
            print(f"\n[{cond}] 결과 없음")
            continue
        n = len(rows)
        print(f"\n[{cond}]  (n={n})")
        print(f"  {'지표':22s} {'평균':>10s} {'표준편차':>10s} {'변동계수(CV%)':>14s}")
        print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*14}")
        cond_stats = {}
        for key, label in FORMAT_METRIC_LABELS.items():
            vals = [r["format_first"][key] for r in rows]
            mean, std, cv = _mean_std_cv(vals)
            cond_stats[key] = (mean, std, cv)
            print(f"  {label:22s} {mean:10.1f} {std:10.1f} {cv:13.1f}%")
        stats_by_cond[cond] = cond_stats

        pass_rate = sum(r["passed"] for r in rows) / n
        avg_score = sum(r["average_score"] for r in rows) / n
        print(f"\n  (참고) 합격률 {pass_rate*100:.1f}% / 평균 CSM 점수 {avg_score:.2f}")

    # CV 낮은 쪽(더 일정한 쪽) 요약
    if len(stats_by_cond) == 2:
        print("\n" + "-" * 78)
        print("  CV(변동계수) 비교 — 값이 낮을수록 해당 조건에서 더 일정하게 생성됨")
        print("-" * 78)
        on, off = stats_by_cond.get("few_shot_on"), stats_by_cond.get("few_shot_off")
        if on and off:
            for key, label in FORMAT_METRIC_LABELS.items():
                cv_on, cv_off = on[key][2], off[key][2]
                winner = "few_shot_on" if cv_on < cv_off else ("few_shot_off" if cv_off < cv_on else "동률")
                print(f"  {label:22s} on={cv_on:6.1f}%  off={cv_off:6.1f}%  -> 더 일정: {winner}")

    print("\n" + "=" * 78)


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

            fm = record["format_first"]
            logger.info(
                f"  -> {'PASS' if record['passed'] else 'FAIL'} "
                f"글자수(원본) {fm['char_count']} / 문단 {fm['paragraph_count']} / 문장 {fm['sentence_count']} "
                f"/ {elapsed:.0f}초"
            )

            # 장시간 반복 실행 시 CUDA 메모리 파편화를 줄이기 위해 매 실행 후 캐시 정리
            try:
                import gc
                import torch
                del result
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    logger.info(f"전체 완료 -> {args.output}")
    print_summary(results)


if __name__ == "__main__":
    main()
