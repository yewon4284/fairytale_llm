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

    all_conditions = sorted(set(r["condition"] for r in results)) or CONDITIONS
    stats_by_cond = {}
    for cond in all_conditions:
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

        scored_rows = [r for r in rows if r.get("average_score") is not None]
        if scored_rows:
            pass_rate = sum(bool(r["passed"]) for r in scored_rows) / len(scored_rows)
            avg_score = sum(r["average_score"] for r in scored_rows) / len(scored_rows)
            print(f"\n  (참고) 합격률 {pass_rate*100:.1f}% / 평균 CSM 점수 {avg_score:.2f}  (평가 데이터 있는 n={len(scored_rows)})")
        else:
            print("\n  (참고) CSM 평가 데이터 없음 (수동 임포트분)")

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


def print_corpus_by_category(corpus_dir):
    """corpus_dir 안 동화들을 classification별로 묶어 형식 CV를 비교하고,
    어느 카테고리가 형식적으로 가장 일정한지 랭킹을 매긴다. GPU/모델 로딩 불필요."""
    import glob
    from collections import defaultdict
    from src.data_loader import story_to_text

    by_cat = defaultdict(list)
    for fp in sorted(glob.glob(os.path.join(corpus_dir, "*.json"))):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        cls = d.get("classification") or "미분류"
        text = story_to_text(d)
        if text.strip():
            by_cat[cls].append(text)

    if not by_cat:
        print(f"'{corpus_dir}'에서 동화를 찾지 못했습니다.")
        return

    print("\n" + "=" * 100)
    print(f"  카테고리별 형식 일관성(CV) 비교 — {corpus_dir}")
    print("=" * 100)

    cat_stats = {}
    for cls, texts in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        metrics_list = [compute_format_metrics(t) for t in texts]
        stats = {}
        for key in FORMAT_METRIC_LABELS:
            vals = [m[key] for m in metrics_list]
            stats[key] = _mean_std_cv(vals)
        cat_stats[cls] = stats

        print(f"\n[{cls}]  (n={len(texts)})")
        print(f"  {'지표':22s} {'평균':>10s} {'CV(%)':>8s}")
        for key, label in FORMAT_METRIC_LABELS.items():
            mean, std, cv = stats[key]
            print(f"  {label:22s} {mean:10.1f} {cv:7.1f}%")

    print("\n" + "-" * 100)
    print("  카테고리별 종합 랭킹 (6개 형식 지표 CV의 평균 — 낮을수록 형식이 더 일정함)")
    print("-" * 100)
    ranking = []
    for cls, stats in cat_stats.items():
        avg_cv = statistics.mean(stats[key][2] for key in FORMAT_METRIC_LABELS)
        ranking.append((cls, avg_cv, len(by_cat[cls])))
    ranking.sort(key=lambda x: x[1])
    for i, (cls, avg_cv, n) in enumerate(ranking, 1):
        mark = "  <- 가장 일정함" if i == 1 else ""
        print(f"  {i}. {cls:14s}  평균 CV {avg_cv:6.1f}%   (n={n}){mark}")

    print("\n  주의: n이 작은 카테고리(50~60편대)는 CV 추정이 상대적으로 덜 안정적일 수 있습니다.")
    print("=" * 100)
    return ranking


def build_fewshot_text(dir_path, n=2, seed=42, classification=None):
    """dir_path 안 JSON 동화들에서 (classification 지정 시 그것만 필터링 후) n편을 seed 고정으로
    무작위 추출해 퓨샷 프롬프트 텍스트로 합친다. main.py의 load_few_shot()과 동일한 포맷."""
    import glob
    import random as _random
    from src.data_loader import story_to_text

    candidates = []
    for fp in sorted(glob.glob(os.path.join(dir_path, "*.json"))):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        if classification and d.get("classification") != classification:
            continue
        text = story_to_text(d)
        if text.strip():
            candidates.append((d.get("title", ""), text))

    if n is not None and n < len(candidates):
        rnd = _random.Random(seed)
        candidates = rnd.sample(candidates, n)

    lines = []
    for i, (title, text) in enumerate(candidates, 1):
        header = f"[참고 동화 {i}] {title}" if title else f"[참고 동화 {i}]"
        lines.append(f"{header}\n{text}")
    return "\n\n".join(lines), len(candidates)


def load_corpus_stories(corpus_dir, classification=None, exclude_filenames=None):
    """corpus_dir 안의 JSON 동화들을 읽어 (텍스트, 파일명) 리스트로 반환.
    classification 지정 시 해당 분류만, exclude_filenames 지정 시 그 파일명들은 제외."""
    import glob
    from src.data_loader import story_to_text

    exclude_filenames = exclude_filenames or set()
    texts, names = [], []
    for fp in sorted(glob.glob(os.path.join(corpus_dir, "*.json"))):
        fname = os.path.basename(fp)
        if fname in exclude_filenames:
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        if classification and d.get("classification") != classification:
            continue
        text = story_to_text(d)
        if text.strip():
            texts.append(text)
            names.append(fname)
    return texts, names


def import_manual_batch(path, output_path):
    """수동으로 생성된 [{id, topic, 기본, 퓨샷}, ...] 형식 JSON을 ab_results.json에 합친다.
    (세이프가드/Solar 평가 없이 만들어진 데이터라 average_score/passed는 None으로 남기고
    형식 지표(format_first/format_final)만 채워서 --summary, --compare-corpus에서 쓸 수 있게 한다.)
    GPU/모델 로딩 불필요."""
    with open(path, "r", encoding="utf-8") as f:
        manual = json.load(f)

    results = load_existing_results(output_path)
    existing_keys = {(r["topic_id"], r["condition"]) for r in results}

    added = 0
    for item in manual:
        tid = item["id"]
        topic = item.get("topic", "")
        pairs = [("few_shot_off", item.get("기본", "")), ("few_shot_on", item.get("퓨샷", ""))]
        for condition, text in pairs:
            if not text.strip():
                continue
            if (tid, condition) in existing_keys:
                logger.warning(f"topic {tid} ({condition}) 이미 존재 — 건너뜀 (덮어쓰려면 ab_results.json에서 먼저 지우세요)")
                continue
            fm = compute_format_metrics(text)
            record = {
                "topic_id": tid,
                "condition": condition,
                "request": topic,
                "passed": None,
                "total_attempts": None,
                "best_attempt": None,
                "average_score": None,
                "min_score": None,
                "scores": {k: None for k in SCORE_KEYS},
                "elapsed_sec": None,
                "format_first": fm,
                "format_final": fm,
                "first_story": text,
                "final_story": text,
                "source": "manual_import",
            }
            results.append(record)
            existing_keys.add((tid, condition))
            added += 1

    save_results(output_path, results)
    logger.info(f"{added}개 레코드 추가 완료 -> {output_path} (전체 {len(results)}개)")


def print_compare_to_corpus(results, corpus_dir, classification, exclude_dir):
    """실제 동화 코퍼스(예: all_data 441편)의 형식 지표와 생성 결과(on/off)를 비교한다.
    exclude_dir에 들어있는 파일명(퓨샷으로 이미 쓴 것들)은 코퍼스에서 제외해서
    '모델이 안 본 진짜 동화'와의 비교가 되게 한다. GPU/모델 로딩 불필요."""
    exclude_filenames = set()
    if exclude_dir and os.path.isdir(exclude_dir):
        exclude_filenames = {f for f in os.listdir(exclude_dir) if f.endswith(".json")}

    texts, names = load_corpus_stories(corpus_dir, classification or None, exclude_filenames)
    if not texts:
        print(f"'{corpus_dir}'에서 조건(classification={classification!r})에 맞는 동화를 찾지 못했습니다.")
        return

    metrics_list = [compute_format_metrics(t) for t in texts]
    corpus_avg, corpus_std = {}, {}
    for key in FORMAT_METRIC_LABELS:
        vals = [m[key] for m in metrics_list]
        corpus_avg[key] = statistics.mean(vals)
        corpus_std[key] = statistics.stdev(vals) if len(vals) > 1 else 0.0

    label = classification or "전체(분류무관)"
    excl_note = f", 퓨샷 {len(exclude_filenames)}편 제외" if exclude_filenames else ""
    print("\n" + "=" * 96)
    print(f"  실제 동화 코퍼스 비교 — {corpus_dir} / {label} (n={len(texts)}{excl_note})")
    print("=" * 96)
    for key, lbl in FORMAT_METRIC_LABELS.items():
        c, s = corpus_avg[key], corpus_std[key]
        cv = (s / c * 100) if c else 0
        print(f"  {lbl:22s} 평균 {c:8.1f}   표준편차 {s:8.1f}   CV {cv:6.1f}%")

    conds = sorted(set(r["condition"] for r in results))
    if not conds:
        print("\nab_results.json에 생성 결과가 없습니다. 먼저 run_ab_test.py를 실행하세요.")
        return

    print("\n" + "-" * 96)
    print("  코퍼스 대비 생성 결과(attempt 1) 비교  — %차이 = (생성평균-코퍼스평균)/코퍼스평균 x 100")
    print("-" * 96)
    col_w = max(10, 60 // max(1, len(conds)))
    header = f"  {'지표':20s} {'코퍼스평균':>10s} {'코퍼스CV':>9s}"
    for cond in conds:
        header += f" {cond+'평균':>{col_w}s} {cond+'차이':>{col_w}s}"
    print(header)
    for key, lbl in FORMAT_METRIC_LABELS.items():
        c = corpus_avg[key]
        cv = (corpus_std[key] / c * 100) if c else 0
        row = f"  {lbl:20s} {c:10.1f} {cv:8.1f}%"
        for cond in conds:
            vals = [r["format_first"][key] for r in results if r["condition"] == cond]
            mean = statistics.mean(vals) if vals else 0
            diff = (mean - c) / c * 100 if c else 0
            row += f" {mean:{col_w}.1f} {diff:+{col_w-1}.1f}%"
        print(row)
    print("\n  -> 차이의 절대값이 작을수록 실제(held-out) 동화 형식에 더 가깝다는 뜻입니다.")
    print("=" * 96)


def print_compare_to_reference(results, few_shot_n):
    """퓨샷으로 준 원본 참고 동화 자체의 형식 지표와, 생성 결과(on/off) 평균을 비교한다.
    GPU/모델 로딩 없이 data_sorted만 읽어서 계산 (가벼움)."""
    from src.data_loader import get_reference_stories

    refs = get_reference_stories(n=few_shot_n, classification="의사소통")
    if not refs:
        print("참고 동화를 불러오지 못했습니다 (data_sorted 확인 필요).")
        return

    print("\n" + "=" * 90)
    print("  퓨샷 참고 동화(원본) vs 생성 결과 형식 비교")
    print("=" * 90)

    ref_metrics_list = []
    for i, ref in enumerate(refs, 1):
        text = ref.get("text", "").strip()
        m = compute_format_metrics(text)
        ref_metrics_list.append(m)
        print(f"\n[참고 동화 {i}] {ref.get('title', '(제목 없음)')}")
        for key, label in FORMAT_METRIC_LABELS.items():
            print(f"  {label}: {m[key]}")

    ref_avg = {
        key: statistics.mean(m[key] for m in ref_metrics_list)
        for key in FORMAT_METRIC_LABELS
    }
    print(f"\n[참고 동화 평균]  (n={len(ref_metrics_list)})")
    for key, label in FORMAT_METRIC_LABELS.items():
        print(f"  {label}: {ref_avg[key]:.1f}")

    on_rows = [r for r in results if r["condition"] == "few_shot_on"]
    off_rows = [r for r in results if r["condition"] == "few_shot_off"]
    if not on_rows and not off_rows:
        print("\nab_results.json에 생성 결과가 없습니다. 먼저 run_ab_test.py를 실행하세요.")
        return

    print("\n" + "-" * 90)
    print("  참고 동화 대비 생성 결과(attempt 1) 비교  — %차이 = (생성평균-참고동화)/참고동화 x 100")
    print("-" * 90)
    header = f"  {'지표':20s} {'참고동화':>10s} {'퓨샷ON평균':>12s} {'ON 차이':>10s} {'퓨샷OFF평균':>12s} {'OFF 차이':>10s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key, label in FORMAT_METRIC_LABELS.items():
        r = ref_avg[key]
        on_vals = [row["format_first"][key] for row in on_rows]
        off_vals = [row["format_first"][key] for row in off_rows]
        on_mean = statistics.mean(on_vals) if on_vals else 0
        off_mean = statistics.mean(off_vals) if off_vals else 0
        on_diff = (on_mean - r) / r * 100 if r else 0
        off_diff = (off_mean - r) / r * 100 if r else 0
        print(
            f"  {label:20s} {r:10.1f} {on_mean:12.1f} {on_diff:+9.1f}% {off_mean:12.1f} {off_diff:+9.1f}%"
        )

    print("\n  -> ON 차이의 절대값이 OFF 차이의 절대값보다 작을수록,")
    print("     퓨샷이 실제로 참고 동화의 형식 쪽으로 생성 결과를 끌어당겼다는 뜻입니다.")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", default="test_topics.json")
    parser.add_argument("--output", default="ab_results.json")
    parser.add_argument("--limit", type=int, default=None, help="앞에서 N개 주제만 테스트")
    parser.add_argument("--generator", default="1.5-8b", choices=["nano", "1.5-8b"])
    parser.add_argument("--no-safeguard", action="store_true")
    parser.add_argument("--few-shot-n", type=int, default=2)
    parser.add_argument("--summary", action="store_true", help="실행 없이 기존 결과만 집계")
    parser.add_argument("--compare-fewshot", action="store_true",
                         help="실행 없이 참고 동화 원본(2편)과 생성 결과 형식을 비교 (GPU 불필요)")
    parser.add_argument("--compare-corpus", action="store_true",
                         help="실행 없이 실제 동화 코퍼스(예: all_data)와 생성 결과 형식을 비교 (GPU 불필요)")
    parser.add_argument("--corpus-dir", default="all_data", help="코퍼스 비교에 쓸 폴더")
    parser.add_argument("--corpus-classification", default="의사소통",
                         help="코퍼스에서 필터링할 classification. 빈 문자열이면 전체(분류 무관)")
    parser.add_argument("--exclude-dir", default="data_sorted",
                         help="코퍼스에서 제외할 파일명이 들어있는 폴더 (퓨샷으로 이미 쓴 파일 제외용)")
    parser.add_argument("--import-manual", default=None,
                         help="[{id, topic, 기본, 퓨샷}, ...] 형식 JSON 파일 경로. "
                              "세이프가드/Solar 평가 없이 만든 결과를 ab_results.json에 형식 지표만 계산해서 합친다. GPU 불필요")
    parser.add_argument("--compare-categories", action="store_true",
                         help="corpus-dir 안 동화를 classification별로 묶어 형식 CV 랭킹을 매김 (GPU 불필요)")
    parser.add_argument("--extra-condition", default=None,
                         help="세 번째 조건 이름 (예: few_shot_mixed). --extra-fewshot-dir와 함께 사용")
    parser.add_argument("--extra-fewshot-dir", default=None,
                         help="세 번째 조건의 퓨샷 소스 폴더 (예: data_sorted_mixed)")
    args = parser.parse_args()

    if args.import_manual:
        import_manual_batch(args.import_manual, args.output)
        return

    if args.compare_categories:
        print_corpus_by_category(args.corpus_dir)
        return

    if args.compare_fewshot:
        results = load_existing_results(args.output)
        print_compare_to_reference(results, args.few_shot_n)
        return

    if args.compare_corpus:
        results = load_existing_results(args.output)
        print_compare_to_corpus(results, args.corpus_dir, args.corpus_classification, args.exclude_dir)
        return

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
    conditions = list(CONDITIONS)

    if args.extra_condition and args.extra_fewshot_dir:
        extra_text, extra_n = build_fewshot_text(args.extra_fewshot_dir, n=args.few_shot_n)
        logger.info(f"[{args.extra_condition}] 퓨샷 소스: {args.extra_fewshot_dir} (풀 {extra_n}편 중 {args.few_shot_n}편 샘플, {len(extra_text)}자)")
        if not extra_text:
            logger.warning(f"{args.extra_fewshot_dir}에서 퓨샷 텍스트를 만들지 못했습니다.")
        pipelines[args.extra_condition] = FairyTalePipeline(
            generator, safeguard, evaluator, few_shot_text=extra_text
        )
        conditions.append(args.extra_condition)

    results = load_existing_results(args.output)
    done = {(r["topic_id"], r["condition"]) for r in results}

    total_runs = len(topics) * len(conditions)
    run_idx = len(done)

    for topic in topics:
        tid = topic["id"]
        request = topic["topic"]

        for condition in conditions:
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
