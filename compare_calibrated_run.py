"""
compare_calibrated_run.py

두 가지 비교 모드를 지원한다.

[모드 1: --baseline (jsonl) + --calibrated (json)] — 기존 방식
    kanana_solar_eval.jsonl(팀 벤치마크, ground_truth 포함)의 predicted 필드를
    BEFORE로, eval_test_dataset.py 결과를 AFTER로 비교.
    ⚠ 주의: kanana_solar_eval.jsonl의 predicted는 decision_source=kanana_force인
    101건에 대해 "구 하드포싱(S3/S5/S6 태깅 시 Solar 검토 없이 무조건 UNSAFE)"
    파이프라인으로 만들어졌을 가능성이 높다(2026-08-16 실측: 이 모드로 비교하면
    재현율이 91%대로 비정상적으로 높게 나오는데, 하드포싱은 재현율을 인위적으로
    부풀리는 효과가 있음). 즉 이 모드는 "재보정 어댑터"뿐 아니라 "하드포싱 제거"
    "Solar 프롬프트 개정"까지 여러 변화가 동시에 섞인 비교라 어댑터 단독 효과를
    보여주지 못한다. 진짜 어댑터 효과를 보려면 모드 2를 쓸 것.

[모드 2: --ground-truth (jsonl, 라벨만 사용) + --before (json) + --after (json)]
    BEFORE/AFTER를 둘 다 "현재 evaluator.py + eval_test_dataset.py"로, 어댑터만
    바꿔서 돌린 결과로 준다 (예: --before는 --adapter-path 없이, --after는
    --adapter-path finetune/kanana-calibration-adapter/final_adapter로).
    ground-truth 파일은 라벨(ground_truth)만 가져오고 그 predicted/flagged는
    쓰지 않는다 — 파이프라인 로직은 두 실행 모두 동일하게 고정하고 어댑터
    유무만 다르게 하는, 진짜 통제된 비교.

사용법:
    # 모드 1 (구 방식, 참고용)
    python compare_calibrated_run.py --baseline kanana_solar_eval.jsonl --calibrated eval_results_calibrated.json

    # 모드 2 (권장) — 먼저 아래 두 실행을 현재 코드로 만들어야 함:
    #   python eval_test_dataset.py --include-fewshot --output eval_results_base.json
    #   python eval_test_dataset.py --include-fewshot --adapter-path finetune/kanana-calibration-adapter/final_adapter --output eval_results_calibrated.json
    python compare_calibrated_run.py --ground-truth kanana_solar_eval.jsonl \
        --before eval_results_base.json --after eval_results_calibrated.json
"""

import argparse
import json
import sys

from eval_compare import confusion_metrics, mcnemar_exact, norm_label


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return {r.get("file") or r.get("filename"): r for r in rows}


def load_eval_json(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return {r["filename"]: r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=None,
                     help="[모드 1] kanana_solar_eval.jsonl 등 predicted+ground_truth 포함 jsonl")
    ap.add_argument("--calibrated", default=None,
                     help="[모드 1] eval_test_dataset.py 결과 json (재보정 어댑터 적용)")
    ap.add_argument("--ground-truth", default=None,
                     help="[모드 2] ground_truth 라벨만 가져올 jsonl (predicted/flagged는 무시)")
    ap.add_argument("--before", default=None, help="[모드 2] 어댑터 없이 현재 코드로 돌린 eval_test_dataset.py 결과 json")
    ap.add_argument("--after", default=None, help="[모드 2] 재보정 어댑터로 현재 코드로 돌린 eval_test_dataset.py 결과 json")
    args = ap.parse_args()

    mode2 = args.ground_truth and args.before and args.after
    mode1 = args.baseline and args.calibrated
    if not (mode1 or mode2):
        print("모드 1: --baseline + --calibrated 를 주거나\n모드 2: --ground-truth + --before + --after 를 주세요.")
        sys.exit(1)

    if mode2:
        print("[모드 2: 통제된 비교 — 어댑터만 다르고 파이프라인 코드는 동일]")
        gt_source = load_jsonl(args.ground_truth)
        before_source = load_eval_json(args.before)
        after_source = load_eval_json(args.after)
        matched = []
        unmatched = []
        for fname, gt_rec in gt_source.items():
            b_rec = before_source.get(fname)
            a_rec = after_source.get(fname)
            if b_rec is None or a_rec is None:
                unmatched.append(fname)
                continue
            matched.append({
                "filename": fname,
                "title": gt_rec.get("title", ""),
                "ground_truth": gt_rec.get("ground_truth"),
                "before_predicted": b_rec.get("verdict"),
                "after_predicted": a_rec.get("verdict"),
                "before_flagged_categories": b_rec.get("flagged_categories") or [],
                "after_flagged_categories": a_rec.get("flagged_categories") or [],
                "decision_source": gt_rec.get("decision_source"),
            })
        if unmatched:
            print(f"  [경고] ground-truth에는 있지만 before/after 결과에 없는 파일 {len(unmatched)}개 (건너뜀): "
                  f"{unmatched[:5]}{'...' if len(unmatched) > 5 else ''}")
        print(f"ground_truth {len(gt_source)}편 / before {len(before_source)}편 / after {len(after_source)}편")
    else:
        print("[모드 1: 구 방식 — kanana_solar_eval.jsonl의 predicted를 BEFORE로 사용, 여러 변화 혼재 가능]")
        baseline = load_jsonl(args.baseline)
        calibrated = load_eval_json(args.calibrated)
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
                "before_flagged_categories": base_rec.get("flagged_categories") or [],
                "after_flagged_categories": cal_rec.get("flagged_categories") or [],
                "decision_source": base_rec.get("decision_source"),
            })
        if unmatched:
            print(f"  [경고] baseline에는 있지만 calibrated 결과에 없는 파일 {len(unmatched)}개 (건너뜀): "
                  f"{unmatched[:5]}{'...' if len(unmatched) > 5 else ''}")

    n = len(matched)
    print(f"\nground_truth와 매칭되어 실제 비교 가능한 편수: {n}\n")
    if n == 0:
        print("비교할 매칭 레코드가 없습니다. --baseline/--calibrated 파일명을 확인하세요.")
        sys.exit(1)

    # ── 원래 문제였던 "오탐(FP) 44건"이 실제로 해소됐는지 직접 확인 ──────────────
    # decision_source=='kanana_force' + ground_truth=='safe' 인 레코드가 바로
    # 하드포싱 때문에 억지로 UNSAFE 처리됐던, 이 프로젝트가 애초에 고치려던 그 대상.
    # 재현율 손실과는 완전히 별개 지표이므로 반드시 따로 확인해야 함.
    fp_target = [r for r in matched
                 if r.get("decision_source") == "kanana_force"
                 and norm_label(r["ground_truth"]) == "SAFE"]
    if fp_target:
        print("=" * 70)
        print(f"  원래 오탐(FP) 대상 {len(fp_target)}건 — 하드포싱으로 억지 UNSAFE 처리됐던 SAFE 동화들")
        print("=" * 70)
        fp_resolved_after = 0
        for r in fp_target:
            a = norm_label(r["after_predicted"])
            ok = (a == "SAFE")
            fp_resolved_after += ok
            print(f"    {r['filename']} {r['title']}: AFTER={r['after_predicted']} "
                  f"{'✅ 해소됨' if ok else '❌ 아직도 UNSAFE'}")
        print(f"\n  FP 해소: {fp_resolved_after}/{len(fp_target)} "
              f"({fp_resolved_after/len(fp_target)*100:.1f}%)")
        print("  (이게 낮으면 재현율 트레이드와 별개로 원래 목표였던 오탐 문제 자체가 안 풀린 것)\n")

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
        print(f"    {r['filename']} {r['title']}: gt={r['ground_truth']} before={r['before_predicted']} -> after={r['after_predicted']}"
              f" | before_flagged={r['before_flagged_categories']}")

    print(f"\n  퇴행한 사례(재현율 손실 등, {len(flips_regressed)}건):")
    for r in flips_regressed:
        print(f"    {r['filename']} {r['title']}: gt={r['ground_truth']} before={r['before_predicted']} -> after={r['after_predicted']}"
              f" | before_flagged={r['before_flagged_categories']}")

    # 재보정이 목표로 삼은 카테고리(S3/S5/S6)와 무관한 곳에서 손실이 났는지 진단.
    # 목표 카테고리 안에서만 손실이 났다면 "의도한 트레이드오프가 너무 강했다"는 뜻이고,
    # S1/S2/S4/S7처럼 손댄 적 없는 카테고리에서도 손실이 났다면 파인튜닝이 좁은 도메인
    # 보정을 넘어 전반적으로 더 관대해지는 방향으로 모델을 밀어버렸다(의도치 않은 표류)는 뜻.
    CALIBRATION_TARGET = {"S3", "S5", "S6"}
    in_target = [r for r in flips_regressed
                 if set(r["before_flagged_categories"]) & CALIBRATION_TARGET]
    outside_target = [r for r in flips_regressed
                       if not (set(r["before_flagged_categories"]) & CALIBRATION_TARGET)]
    print("\n" + "=" * 70)
    print("  퇴행 원인 진단: 재보정 대상(S3/S5/S6) 안에서 손실 vs 밖에서 손실")
    print("=" * 70)
    print(f"  재보정 대상 카테고리(S3/S5/S6)가 걸려있던 퇴행: {len(in_target)}건 "
          f"(의도된 트레이드오프가 너무 강했을 가능성)")
    print(f"  재보정과 무관한 카테고리(S1/S2/S4/S7 등)에서의 퇴행: {len(outside_target)}건 "
          f"(파인튜닝이 목표 밖 카테고리까지 관대하게 만든 표류일 가능성)")
    if outside_target:
        print("  무관 카테고리 퇴행 목록:")
        for r in outside_target:
            print(f"    {r['filename']} {r['title']}: before_flagged={r['before_flagged_categories']}")

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
