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
    python run_ab_test.py                          # 전부, 두 조건(few_shot_on/off)
    python run_ab_test.py --limit 3                # 앞 3개 주제만 (동작 확인용)
    python run_ab_test.py --no-safeguard            # 세이프가드 생략 (빠른 디버그용)
    python run_ab_test.py --summary                 # 실행 없이 기존 결과만 집계해서 출력

    # 추가 조건(카테고리별 퓨샷 등)을 여러 개 동시에 돌리기 — 순서대로 짝지어짐
    python run_ab_test.py \
        --extra-condition cat_의사소통 --extra-fewshot-dir data_sorted_cat_의사소통 \
        --extra-condition cat_예술경험 --extra-fewshot-dir data_sorted_cat_예술경험

    # 원본 동화 기준값(카테고리 균등가중, CV 높은 카테고리는 중앙값) 계산만
    python run_ab_test.py --corpus-baseline --corpus-dir all_data

    # 조건별 |원본기준값-생성값| paired Wilcoxon 검정 + FDR 보정 (기준: few_shot_off)
    python run_ab_test.py --paired-test --corpus-dir all_data
"""

import argparse
import json
import logging
import math
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

# 큰따옴표로 감싸진 구간을 대사로 간주
DIALOGUE_RE = re.compile(r'"([^"]*)"')

# 형식 지표로 집계할 필드와 표시 이름
FORMAT_METRIC_LABELS = {
    "char_count": "글자수(공백제외)",
    "char_count_ws": "글자수(공백포함)",
    "word_count": "단어수(어절)",
    "sentence_count": "문장수",
    "paragraph_count": "문단수",
    "avg_sentence_len": "문장당 평균 글자수",
    "avg_paragraph_len": "문단당 평균 글자수",
    "dialogue_ratio": "대사 비중(%)",
}

# "원본에 가까운가" 비교의 핵심 지표 — 개수 그대로 쓰면 총 길이 차이로 왜곡되므로
# 문단당 글자수(비율)와 대사 비중(비율) 두 개만 원본-대비 핵심 비교에 쓴다.
CORE_COMPARISON_METRICS = ("avg_paragraph_len", "dialogue_ratio")


def load_topics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_results(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            results = json.load(f)
        if _backfill_format_metrics(results):
            save_results(path, results)
            logger.info(f"{path}: 예전 레코드에 누락된 형식 지표(대사 비중 등)를 다시 계산해 채워 넣었습니다.")
        return results
    return []


def _backfill_format_metrics(results) -> bool:
    """예전(대사 비중 지표 추가 이전) 레코드에는 format_first/format_final에
    dialogue_ratio 등 새 필드가 없다. 저장된 first_story/final_story 텍스트에서
    다시 계산해 채워 넣는다. 뭔가 채워 넣었으면 True를 반환."""
    changed = False
    for r in results:
        for field_key, text_key in (("format_first", "first_story"), ("format_final", "final_story")):
            fm = r.get(field_key)
            text = r.get(text_key)
            if fm is not None and text and "dialogue_ratio" not in fm:
                fm.update(compute_format_metrics(text))
                changed = True
    return changed


def save_results(path, results):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def compute_dialogue_chars(story: str) -> int:
    """큰따옴표 " " 안의 텍스트(대사)만 뽑아 공백/줄바꿈 제외 글자수를 센다."""
    dialogue_text = "".join(DIALOGUE_RE.findall(story))
    return len(dialogue_text.replace(" ", "").replace("\n", ""))


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

    dialogue_chars = compute_dialogue_chars(story)
    dialogue_ratio = round(dialogue_chars / char_count * 100, 1) if char_count else 0.0

    return {
        "char_count": char_count,
        "char_count_ws": char_count_ws,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "avg_sentence_len": round(char_count / sentence_count, 1),
        "avg_paragraph_len": round(char_count / paragraph_count, 1),
        "dialogue_chars": dialogue_chars,
        "dialogue_ratio": dialogue_ratio,
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


def load_stories_by_category(corpus_dir):
    """corpus_dir 안 JSON 동화들을 classification별로 묶어 {classification: [text, ...]}로 반환."""
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
    return by_cat


def print_corpus_by_category(corpus_dir):
    """corpus_dir 안 동화들을 classification별로 묶어 형식 CV를 비교하고,
    어느 카테고리가 형식적으로 가장 일정한지 랭킹을 매긴다. GPU/모델 로딩 불필요."""
    by_cat = load_stories_by_category(corpus_dir)

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


def compute_corpus_baseline(corpus_dir, metrics=CORE_COMPARISON_METRICS, cv_threshold=60.0):
    """'원본 동화' 기준값을 카테고리 구성 불균형을 통제해서 계산한다.

    절차 (결과를 보기 전에 고정한 규칙):
      1) 카테고리별 대표값 산출 — 그 카테고리의 CV가 cv_threshold(%)를 넘으면 중앙값,
         아니면 평균을 대표값으로 쓴다 (CV 높은 카테고리의 극단값이 대표값을 왜곡하는 것을 방지).
      2) 카테고리 대표값들을 '균등가중'으로 평균해서 최종 기준값을 만든다
         (표본수 많은 카테고리, 예: 자연탐구 175편이 기준을 독식하지 않도록).

    반환: (baseline, detail, pooled)
      baseline: {metric: 최종 기준값}
      detail:   {metric: {classification: {n, mean, median, cv, rep, method}}}
      pooled:   {metric: 카테고리 구분 없이 전체를 그냥 풀링했을 때의 평균} (비교/검증용)
    """
    by_cat = load_stories_by_category(corpus_dir)
    baseline, detail, pooled = {}, {}, {}

    for metric in metrics:
        detail[metric] = {}
        reps = []
        all_vals = []
        for cls, texts in by_cat.items():
            vals = [compute_format_metrics(t)[metric] for t in texts]
            all_vals.extend(vals)
            mean, std, cv = _mean_std_cv(vals)
            median = statistics.median(vals) if vals else 0.0
            use_median = cv > cv_threshold
            rep = median if use_median else mean
            detail[metric][cls] = {
                "n": len(vals), "mean": mean, "median": median, "cv": cv,
                "rep": rep, "method": "중앙값" if use_median else "평균",
            }
            reps.append(rep)
        baseline[metric] = statistics.mean(reps) if reps else 0.0
        pooled[metric] = statistics.mean(all_vals) if all_vals else 0.0

    return baseline, detail, pooled


def print_corpus_baseline(corpus_dir, cv_threshold=60.0):
    """원본 동화 기준값(카테고리 균등가중, CV 높은 카테고리는 중앙값 사용)을 계산해서 출력.
    GPU/모델 로딩 불필요."""
    baseline, detail, pooled = compute_corpus_baseline(corpus_dir, cv_threshold=cv_threshold)

    print("\n" + "=" * 100)
    print(f"  원본 동화 기준값 산출 — {corpus_dir}  (카테고리 CV > {cv_threshold:.0f}% 이면 중앙값 사용, 아니면 평균)")
    print("=" * 100)
    for metric in CORE_COMPARISON_METRICS:
        label = FORMAT_METRIC_LABELS[metric]
        print(f"\n[{label}]")
        print(f"  {'카테고리':14s} {'n':>4s} {'평균':>10s} {'중앙값':>10s} {'CV(%)':>8s} {'대표값':>10s} {'방식':>6s}")
        for cls, d in sorted(detail[metric].items(), key=lambda x: -x[1]["n"]):
            print(f"  {cls:14s} {d['n']:4d} {d['mean']:10.1f} {d['median']:10.1f} {d['cv']:7.1f}% {d['rep']:10.1f} {d['method']:>6s}")
        print(f"  {'-'*14} {'-'*4} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*6}")
        print(f"  {'균등가중 기준값':14s} {'':>4s} {'':>10s} {'':>10s} {'':>8s} {baseline[metric]:10.1f}")
        print(f"  (참고) 카테고리 구분 없이 {corpus_dir} 전체를 그냥 풀링한 평균: {pooled[metric]:.1f}"
              f"  (차이 {(baseline[metric]-pooled[metric])/pooled[metric]*100:+.1f}%)" if pooled[metric] else "")
    print("\n" + "=" * 100)
    return baseline, detail, pooled


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilcoxon_signed_rank(diffs):
    """paired 표본의 부호순위검정 (정규근사, 연속성 보정). scipy 없이 순수 파이썬 구현.
    diffs: (조건A 값 - 조건B 값) 리스트. 0인 차이는 제외하고 계산.
    반환: {"n": 유효표본수, "z": z통계량, "p": 양측 p값} 또는 표본이 너무 적으면 None."""
    diffs = [d for d in diffs if d != 0]
    n = len(diffs)
    if n < 4:
        return None

    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    abs_sorted = [abs(diffs[i]) for i in order]
    ranks = [0.0] * n
    idx = 0
    while idx < n:
        j = idx
        while j + 1 < n and abs_sorted[j + 1] == abs_sorted[idx]:
            j += 1
        avg_rank = (idx + 1 + j + 1) / 2.0
        for k in range(idx, j + 1):
            ranks[order[k]] = avg_rank
        idx = j + 1

    w_plus = sum(ranks[i] for i in range(n) if diffs[i] > 0)
    w_minus = sum(ranks[i] for i in range(n) if diffs[i] < 0)
    w = min(w_plus, w_minus)
    mean_w = n * (n + 1) / 4.0
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if std_w == 0:
        return None
    z = (w - mean_w + (0.5 if w < mean_w else -0.5)) / std_w
    p = min(1.0, 2 * (1 - _norm_cdf(abs(z))))
    return {"n": n, "w_plus": w_plus, "w_minus": w_minus, "z": z, "p": p}


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR 보정. pvals 리스트(None 허용, 순서 유지)를 받아 같은 순서의 q값 리스트를 반환."""
    indexed = [(i, p) for i, p in enumerate(pvals) if p is not None]
    m = len(indexed)
    if m == 0:
        return [None] * len(pvals)
    indexed_sorted = sorted(indexed, key=lambda x: x[1])
    qvals = {}
    prev_q = 1.0
    for k in range(m - 1, -1, -1):
        i, p = indexed_sorted[k]
        rank = k + 1
        q = min(p * m / rank, prev_q)
        qvals[i] = q
        prev_q = q
    return [qvals.get(i) for i in range(len(pvals))]


def print_paired_stats(results, corpus_dir, cv_threshold=60.0, ref_condition="few_shot_off"):
    """조건별로 '원본 기준값에서 얼마나 벗어났는지(절대편차)'를 주제 단위로 짝지어(paired)
    ref_condition과 비교하는 Wilcoxon 부호순위검정 + BH-FDR 다중비교 보정. GPU/모델 로딩 불필요.

    가설: 퓨샷 조건은 ref_condition(기본: 퓨샷 OFF)보다 원본 기준값에서 벗어난 정도(|생성값-기준값|)가
          더 작다 (= 원본 형식에 더 가깝다).
    """
    baseline, _, _ = compute_corpus_baseline(corpus_dir, cv_threshold=cv_threshold)

    all_conditions = sorted(set(r["condition"] for r in results))
    if ref_condition not in all_conditions:
        print(f"\n기준 조건 '{ref_condition}'의 데이터가 ab_results.json에 없습니다.")
        return
    other_conditions = [c for c in all_conditions if c != ref_condition]
    if not other_conditions:
        print("\n비교할 다른 조건이 없습니다.")
        return

    print("\n" + "=" * 100)
    print(f"  paired 검정 — 원본 기준값 대비 |편차| 비교 (기준 조건: {ref_condition})")
    print(f"  귀무가설: 두 조건의 |생성값-원본기준값| 분포에 차이가 없다")
    print("=" * 100)

    tests = []  # (metric, cond, result_dict or None, median_diff)
    for metric in CORE_COMPARISON_METRICS:
        target = baseline[metric]
        ref_by_tid = {
            r["topic_id"]: r["format_first"][metric]
            for r in results if r["condition"] == ref_condition
        }
        for cond in other_conditions:
            cond_by_tid = {
                r["topic_id"]: r["format_first"][metric]
                for r in results if r["condition"] == cond
            }
            common = sorted(set(ref_by_tid) & set(cond_by_tid))
            diffs = [
                abs(cond_by_tid[t] - target) - abs(ref_by_tid[t] - target)
                for t in common
            ]
            res = wilcoxon_signed_rank(diffs)
            median_diff = statistics.median(diffs) if diffs else None
            tests.append((metric, cond, res, median_diff, len(common)))

    pvals = [t[2]["p"] if t[2] else None for t in tests]
    qvals = bh_fdr(pvals)

    for (metric, cond, res, median_diff, npair), q in zip(tests, qvals):
        label = FORMAT_METRIC_LABELS[metric]
        if res is None:
            print(f"  [{label}] {cond} vs {ref_condition}: 짝지어진 표본 부족(n={npair}) — 검정 생략")
            continue
        direction = "원본에 더 가까움" if median_diff < 0 else ("원본에서 더 멂" if median_diff > 0 else "차이 없음")
        sig = "*" if (q is not None and q < 0.05) else " "
        print(
            f"  [{label}] {cond:14s} vs {ref_condition:14s}  n={res['n']:3d}  "
            f"z={res['z']:+6.2f}  p={res['p']:.4f}  q(FDR)={q:.4f}{sig}  "
            f"median(|편차A|-|편차B|)={median_diff:+7.1f}  -> {cond}가 {direction}"
        )
    print("\n  * = q<0.05 (FDR 보정 후에도 유의). median 값이 음수면 해당 조건이 기준 조건보다 원본에 더 가깝다는 뜻.")
    print(f"  참고: 표본 수가 작으면(n<20~30) 검정력이 낮아 유의하지 않아도 실제 효과가 없다는 뜻은 아닙니다.")
    print("=" * 100)


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
    parser.add_argument("--extra-condition", action="append", default=[],
                         help="추가 조건 이름 (예: cat_의사소통). 여러 번 줄 수 있음. --extra-fewshot-dir와 순서대로 짝지어짐")
    parser.add_argument("--extra-fewshot-dir", action="append", default=[],
                         help="추가 조건의 퓨샷 소스 폴더 (예: data_sorted_cat_의사소통). --extra-condition과 같은 개수/순서")
    parser.add_argument("--corpus-baseline", action="store_true",
                         help="실행 없이 원본 동화 기준값(카테고리 균등가중, CV 높은 카테고리는 중앙값) 계산해서 출력 (GPU 불필요)")
    parser.add_argument("--cv-threshold", type=float, default=60.0,
                         help="카테고리 대표값 산출 시 평균 대신 중앙값을 쓰는 CV(%%) 임계값 (기본 60)")
    parser.add_argument("--paired-test", action="store_true",
                         help="실행 없이 조건별 원본기준값 대비 |편차|를 ref-condition과 paired Wilcoxon 검정 + FDR 보정 (GPU 불필요)")
    parser.add_argument("--ref-condition", default="few_shot_off",
                         help="--paired-test에서 비교 기준으로 삼을 조건 (기본 few_shot_off)")
    args = parser.parse_args()

    if len(args.extra_condition) != len(args.extra_fewshot_dir):
        logger.error("--extra-condition과 --extra-fewshot-dir 개수가 다릅니다 "
                      f"({len(args.extra_condition)} vs {len(args.extra_fewshot_dir)}). 순서대로 짝을 맞춰 주세요.")
        sys.exit(1)

    if args.import_manual:
        import_manual_batch(args.import_manual, args.output)
        return

    if args.compare_categories:
        print_corpus_by_category(args.corpus_dir)
        return

    if args.corpus_baseline:
        print_corpus_baseline(args.corpus_dir, cv_threshold=args.cv_threshold)
        return

    if args.paired_test:
        results = load_existing_results(args.output)
        print_paired_stats(results, args.corpus_dir, cv_threshold=args.cv_threshold, ref_condition=args.ref_condition)
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

    for extra_condition, extra_dir in zip(args.extra_condition, args.extra_fewshot_dir):
        extra_text, extra_n = build_fewshot_text(extra_dir, n=args.few_shot_n)
        logger.info(f"[{extra_condition}] 퓨샷 소스: {extra_dir} (풀 {extra_n}편 중 {args.few_shot_n}편 샘플, {len(extra_text)}자)")
        if not extra_text:
            logger.warning(f"{extra_dir}에서 퓨샷 텍스트를 만들지 못했습니다.")
        pipelines[extra_condition] = FairyTalePipeline(
            generator, safeguard, evaluator, few_shot_text=extra_text
        )
        conditions.append(extra_condition)

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
