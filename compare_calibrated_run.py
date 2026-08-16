"""
compare_calibrated_run.py
eval_test_dataset.py로 새로 돌린 재보정 어댑터 결과(eval_results_calibrated.json, 441편)를
기존 baseline(kanana_solar_eval.jsonl, 421편 — ground_truth + 재보정 전 predicted 포함)과
비교한다.

kanana_solar_eval.jsonl은 사람이 라벨링한 ground_truth를 담고 있는 유일한 파일이므로,
"441편 중 몇 개가 SAFE/UNSAFE냐"만으로는 증명이 안 되고 반드시 이 파일과 join해서
정답 대비 정확도를 봐야 함. 또한 predicted 필드가 재보정 전 시스템의 판정이므로
같은 421편에 대해 before/after paired McNemar 검정이 가능함 (완전한 held-out은 아니지만
현재 갖고 있는 것 중 가장 엄밀한 비교).

주의: eval_results_calibrated.json의 441편 중 20편은 퓨샷 세트라 kanana_solar_eval.jsonl에
ground_truth가 없음 -> 자동으로 매칭 안 되는 건 스킵하고 몇 개 스킵됐는지 출력함.

사용법:
    python compare_calibrated_run.py \
        --baseline kanana_solar_eval.jsonl \
        --calibrated eval_results_calibrated.json
"""

import argparse
import json
import sys

from eval_compare import confusion_metrics, mcnemar_exact, norm_label


def load_baseline(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return {r.get("file") or r.get("filename"): r for r in rows}


def load_calibrated(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return {r["filename"]: r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="kanana_solar_eval.jsonl")
    ap.add_argument("--calibrated", default="eval_results_calibrated.json")
    args = ap.parse_args()

    baseline = load_baseline(args.baseline)
    calibrated = load_calibrated(args.calibrated)

    print(f"baseline(ground_truth 포함) {len(baseline)}편 / calibrated 실행 결과 {len(calibrated)}편")

    matched = []
    unmatched = []
    for fname, base_rec in baseline.items():
        cal_rec = calibrated.get(fname)
        if cal_rec is None:
            unmatched.append(fname)
            continue
        matched.append({
            "filename": fname,
            "title": base_rec.get("title", ""),
            "ground_truth": base_rec.get("ground_truth"),
            "before_predicted": base_rec.get("predicted"),
            "after_predicted": cal_rec.get("verdict"),
        })

    if unmatched:
        print(f"  [경고] baseline에는 있지만 calibrated 결과에 없는 파일 {len(unmatched)}개 (건너뜀): "
              f"{unmatched[:5]}{'...' if len(unmatched) > 5 else ''}")

    n = len(matched)
    print(f"\nground_truth와 매칭되어 실제 비교 가능한 편수: {n}\n")
    if n == 0:
        print("비교할 매칭 레코드가 없습니다. --baseline/--calibrated 파일명을 확인하세요.")
        sys.exit(1)

    print("=" * 70)
    print("  BEFORE(재보정 전 predicted) vs ground_truth")
    print("=" * 70)
    before_metrics = confusion_metrics(matched, "before_predicted", "ground_truth")
    for k, v in before_metrics.items():
        print(f"  {k}: {v if isinstance(v, int) else round(v, 4) if v == v else v}")

    print("\n" + "=" * 70)
    print("  AFTER(재보정 후 predicted) vs ground_truth")
    print("=" * 70)
    after_metrics = confusion_metrics(matched, "after_predicted", "ground_truth")
    for k, v in after_metrics.items():
        print(f"  {k}: {v if isinstance(v, int) else round(v, 4) if v == v else v}")

    print("\n" + "=" * 70)
    print("  Paired McNemar exact test (BEFORE vs AFTER, 같은 421편 기준)")
    print("=" * 70)
    before_correct = []
    after_correct = []
    flips_improved = []   # before 틀림 -> after 맞음
    flips_regressed = []  # before 맞음 -> after 틀림
    for r in matched:
        gt = norm_label(r["ground_truth"])
        b = norm_label(r["before_predicted"])
        a = norm_label(r["after_predicted"])
        if gt is None or b is None or a is None:
            before_correct.append(False)
            after_correct.append(False)
            continue
        b_ok = (b == gt)
        a_ok = (a == gt)
        before_correct.append(b_ok)
        after_correct.append(a_ok)
        if not b_ok and a_ok:
            flips_improved.append(r)
        elif b_ok and not a_ok:
            flips_regressed.append(r)

    b, c, p = mcnemar_exact(after_correct, before_correct)
    print(f"  AFTER만 맞음(개선): {b}건 / BEFORE만 맞음(퇴행): {c}건 / p-value: {p:.4f}")
    if p < 0.05:
        direction = "AFTER가 유의미하게 더 정확함" if b > c else "BEFORE가 유의미하게 더 정확함"
        print(f"  -> p<0.05, 통계적으로 유의미한 차이 있음: {direction}")
    else:
        print("  -> p>=0.05, 통계적으로 유의미한 차이라고 보기 어려움 (표본 수가 작아 흔한 결과)")

    print(f"\n  개선된 사례(오탐 해소 등, {len(flips_improved)}건):")
    for r in flips_improved:
        print(f"    {r['filename']} {r['title']}: gt={r['ground_truth']} before={r['before_predicted']} -> after={r['after_predicted']}")

    print(f"\n  퇴행한 사례(재현율 손실 등, {len(flips_regressed)}건):")
    for r in flips_regressed:
        print(f"    {r['filename']} {r['title']}: gt={r['ground_truth']} before={r['before_predicted']} -> after={r['after_predicted']}")

    print("\n" + "=" * 70)
    print("  요약")
    print("=" * 70)
    print(f"  정확도: BEFORE {before_metrics['acc']:.4f} -> AFTER {after_metrics['acc']:.4f} "
          f"(Δ{after_metrics['acc']-before_metrics['acc']:+.4f})")
    print(f"  정밀도: BEFORE {before_metrics['prec']:.4f} -> AFTER {after_metrics['prec']:.4f} "
          f"(Δ{after_metrics['prec']-before_metrics['prec']:+.4f})")
    print(f"  재현율: BEFORE {before_metrics['rec']:.4f} -> AFTER {after_metrics['rec']:.4f} "
          f"(Δ{after_metrics['rec']-before_metrics['rec']:+.4f})")
    print(f"  F1    : BEFORE {before_metrics['f1']:.4f} -> AFTER {after_metrics['f1']:.4f} "
          f"(Δ{after_metrics['f1']-before_metrics['f1']:+.4f})")
    print(f"  McNemar p-value: {p:.4f}")


if __name__ == "__main__":
    main()
