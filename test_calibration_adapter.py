"""
test_calibration_adapter.py
재보정 학습한 LoRA 어댑터(finetune/kanana-calibration-adapter)가 실제로
오탐(FP)을 줄이면서 정탐(TP) 탐지력은 유지하는지 확인하는 A/B 테스트.

비교 대상:
  - BEFORE: 어댑터 없이 베이스 모델만 (S5_ADAPTER_PATH의 기존 어댑터는 실제 가중치가
    git에 커밋된 적이 없어 지금까지 사실상 항상 베이스 모델로만 동작해왔음 — 2026-08-16
    확인. 그래서 "기존과 비교"는 곧 "베이스 모델과 비교"와 같다.)
  - AFTER : finetune/kanana-calibration-adapter/final_adapter 적용

GPU 메모리 절약을 위해 두 모델을 동시에 올리지 않고, 순차적으로 로딩 -> 평가 ->
메모리 해제 -> 재로딩한다.

GPU 필요 (pod에서 실행).

사용법:
    python test_calibration_adapter.py --eval-jsonl kanana_solar_eval.jsonl --corpus-dir all_data
    python test_calibration_adapter.py --eval-jsonl kanana_solar_eval.jsonl --corpus-dir all_data \
        --new-adapter finetune/kanana-calibration-adapter/final_adapter
"""

import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CATEGORY_PREFIX = {"01": "03_01", "02": "03_02", "03": "03_03", "04": "03_04", "05": "03_05"}
_category_cache = {}
FORCE_CATS = {"S3", "S5", "S6"}


def resolve_filename(short_name, corpus_dir):
    direct = os.path.join(corpus_dir, short_name)
    if os.path.exists(direct):
        return short_name
    base = short_name[:-5] if short_name.endswith(".json") else short_name
    parts = base.split("_")
    if len(parts) != 2 or parts[0] not in CATEGORY_PREFIX:
        return short_name
    cat, idx_str = parts
    try:
        idx = int(idx_str)
    except ValueError:
        return short_name
    prefix = CATEGORY_PREFIX[cat]
    if prefix not in _category_cache:
        _category_cache[prefix] = sorted(
            fn for fn in os.listdir(corpus_dir) if fn.startswith(prefix)
        )
    files = _category_cache[prefix]
    if 1 <= idx <= len(files):
        return files[idx - 1]
    return short_name


def evaluate_all(safeguard, story_to_text, records, corpus_dir):
    """레코드 목록에 대해 세이프가드를 돌려서 {file: set(강제카테고리 중 잡힌 것)}을 반환."""
    results = {}
    for r in records:
        short_name = r.get("file") or r.get("filename")
        fname = resolve_filename(short_name, corpus_dir)
        path = os.path.join(corpus_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            text = story_to_text(json.load(f))
        _, flagged = safeguard.evaluate_story(text)
        cats = {f["category"] for f in flagged} & FORCE_CATS
        results[short_name] = cats
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--corpus-dir", default="all_data")
    ap.add_argument("--new-adapter", default="finetune/kanana-calibration-adapter/final_adapter")
    ap.add_argument("--limit", type=int, default=None, help="FP/TP 각각 앞에서 N건만 (디버그용)")
    args = ap.parse_args()

    from src.data_loader import story_to_text
    from src.safeguard import KananaSafeguard

    recs = []
    with open(args.eval_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    fp_records = [
        r for r in recs
        if r.get("decision_source") == "kanana_force" and r.get("ground_truth") == "safe"
    ]
    tp_records = [
        r for r in recs
        if r.get("decision_source") == "kanana_force" and r.get("ground_truth") == "unsafe"
    ]
    if args.limit:
        fp_records = fp_records[: args.limit]
        tp_records = tp_records[: args.limit]

    fp_forced = {(r.get("file") or r.get("filename")): set(r.get("forced_categories") or [])
                 for r in fp_records}
    tp_forced = {(r.get("file") or r.get("filename")): set(r.get("forced_categories") or [])
                 for r in tp_records}

    print(f"FP(오탐) {len(fp_records)}건 / TP(정탐) {len(tp_records)}건 테스트\n")

    print("=" * 60)
    print("  [BEFORE] 베이스 모델 (어댑터 없음)")
    print("=" * 60)
    safeguard_before = KananaSafeguard(use_s5_adapter=False)
    before_fp = evaluate_all(safeguard_before, story_to_text, fp_records, args.corpus_dir)
    before_tp = evaluate_all(safeguard_before, story_to_text, tp_records, args.corpus_dir)
    del safeguard_before
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    print("\n" + "=" * 60)
    print(f"  [AFTER] 재보정 어댑터 적용: {args.new_adapter}")
    print("=" * 60)
    safeguard_after = KananaSafeguard(use_s5_adapter=True, adapter_path=args.new_adapter)
    after_fp = evaluate_all(safeguard_after, story_to_text, fp_records, args.corpus_dir)
    after_tp = evaluate_all(safeguard_after, story_to_text, tp_records, args.corpus_dir)

    # ── FP 비교 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FP(오탐) 해소 비교")
    print("=" * 60)
    fp_resolved_before = fp_resolved_after = 0
    for fname, orig_cats in fp_forced.items():
        b_cats = before_fp.get(fname, set()) & orig_cats
        a_cats = after_fp.get(fname, set()) & orig_cats
        b_ok = not b_cats
        a_ok = not a_cats
        fp_resolved_before += b_ok
        fp_resolved_after += a_ok
        mark = ""
        if a_ok and not b_ok:
            mark = "  <- 재보정으로 새로 해소됨"
        elif b_ok and not a_ok:
            mark = "  <- ⚠ 재보정 후 오히려 다시 잡힘 (회귀)"
        print(f"  {fname}: BEFORE={'해소' if b_ok else sorted(b_cats)} / "
              f"AFTER={'해소' if a_ok else sorted(a_cats)}{mark}")
    n_fp = len(fp_forced)
    print(f"\n  FP 해소: BEFORE {fp_resolved_before}/{n_fp} ({fp_resolved_before/n_fp*100:.1f}%) "
          f"-> AFTER {fp_resolved_after}/{n_fp} ({fp_resolved_after/n_fp*100:.1f}%)" if n_fp else "")

    # ── TP 비교 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TP(정탐) 유지 비교 (재현율 손실 점검)")
    print("=" * 60)
    tp_preserved_before = tp_preserved_after = 0
    for fname, orig_cats in tp_forced.items():
        b_cats = before_tp.get(fname, set()) & orig_cats
        a_cats = after_tp.get(fname, set()) & orig_cats
        b_ok = b_cats == orig_cats
        a_ok = a_cats == orig_cats
        tp_preserved_before += b_ok
        tp_preserved_after += a_ok
        mark = ""
        if b_ok and not a_ok:
            mark = "  <- ⚠ 재보정 후 놓침 (재현율 손실)"
        elif a_ok and not b_ok:
            mark = "  <- 재보정으로 새로 잡힘"
        print(f"  {fname}: 원래={sorted(orig_cats)} BEFORE={sorted(b_cats)} "
              f"AFTER={sorted(a_cats)}{mark}")
    n_tp = len(tp_forced)
    print(f"\n  TP 유지: BEFORE {tp_preserved_before}/{n_tp} ({tp_preserved_before/n_tp*100:.1f}%) "
          f"-> AFTER {tp_preserved_after}/{n_tp} ({tp_preserved_after/n_tp*100:.1f}%)" if n_tp else "")

    print("\n" + "=" * 60)
    print("  종합 판단")
    print("=" * 60)
    if n_fp and n_tp:
        fp_gain = fp_resolved_after - fp_resolved_before
        tp_loss = tp_preserved_before - tp_preserved_after
        print(f"  FP 추가 해소: {fp_gain}건 / TP 추가 손실: {tp_loss}건")
        if tp_loss > 0 and tp_loss >= fp_gain:
            print("  ⚠ TP 손실이 FP 개선과 비슷하거나 더 큼 — 재보정 어댑터를 그대로 배포하기엔 위험.")
        elif tp_loss > n_tp * 0.1:
            print("  ⚠ TP 손실 비율이 10% 넘음 — 프로덕션 적용 전에 재검토 필요.")
        else:
            print("  FP는 개선되고 TP 손실은 크지 않음 — 프로덕션 적용을 고려할 만함.")


if __name__ == "__main__":
    main()
