"""
test_safeguard_context.py
문맥 윈도우 방식(safeguard.py의 use_context=True)이 실제로 도메인 오탐(FP)을
줄이는지 확인하는 A/B 테스트. 재학습 없이 프롬프트만 바꿔서 즉시 검증 가능.

대상: kanana_solar_eval.jsonl에서 decision_source=kanana_force AND ground_truth=safe
      (사람은 안전하다고 봤는데 카나나가 S3/S5/S6로 강제 태깅한 오탐 사례) 전부 또는 --limit개.

각 사례에 대해:
  - 문맥 없이(use_context=False, 기존 방식) 다시 분류
  - 문맥 포함(use_context=True) 다시 분류
두 결과를 비교해서, 문맥을 넣었을 때 원래 오탐이던 카테고리(S3/S5/S6)가 사라지는지 확인한다.

GPU 필요 (pod에서 실행).

사용법:
    python test_safeguard_context.py --eval-jsonl kanana_solar_eval.jsonl --corpus-dir all_data
    python test_safeguard_context.py --eval-jsonl kanana_solar_eval.jsonl --corpus-dir all_data --limit 10
    python test_safeguard_context.py --eval-jsonl kanana_solar_eval.jsonl --corpus-dir all_data --context-window 2
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# rename_corpus_to_short_ids.py 실행 전(축약 파일명 미적용) pod에서도 동작하도록,
# all_data 카테고리 접두사 기반 역산 매칭을 폴백으로 둔다.
CATEGORY_PREFIX = {"01": "03_01", "02": "03_02", "03": "03_03", "04": "03_04", "05": "03_05"}
_category_cache = {}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--corpus-dir", default="all_data")
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N건만 테스트")
    ap.add_argument("--context-window", type=int, default=1)
    args = ap.parse_args()

    from src.data_loader import story_to_text
    from src.safeguard import KananaSafeguard

    recs = []
    with open(args.eval_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    fp_forced = [
        r for r in recs
        if r.get("decision_source") == "kanana_force" and r.get("ground_truth") == "safe"
    ]
    if args.limit:
        fp_forced = fp_forced[: args.limit]
    print(f"테스트 대상 FP(오탐) 사례: {len(fp_forced)}건\n")

    print("세이프가드 로딩...")
    safeguard = KananaSafeguard()

    resolved_fp = 0     # 문맥 적용 후 강제unsafe 카테고리가 하나도 안 남은 경우
    partial_fp = 0      # 일부는 사라졌지만 일부 남은 경우
    unresolved_fp = 0   # 전혀 안 사라진 경우
    load_failed = 0

    FORCE_CATS = {"S3", "S5", "S6"}

    for r in fp_forced:
        short_name = r.get("file") or r.get("filename")
        forced_cats = set(r.get("forced_categories") or [])
        fname = resolve_filename(short_name, args.corpus_dir)
        path = os.path.join(args.corpus_dir, fname)
        if not os.path.exists(path):
            print(f"[건너뜀] {short_name} — 파일 없음")
            load_failed += 1
            continue

        with open(path, encoding="utf-8") as f:
            text = story_to_text(json.load(f))

        _, flagged_no_ctx = safeguard.evaluate_story(text, use_context=False)
        _, flagged_ctx = safeguard.evaluate_story(
            text, use_context=True, context_window=args.context_window
        )

        cats_no_ctx = {f["category"] for f in flagged_no_ctx} & FORCE_CATS
        cats_ctx = {f["category"] for f in flagged_ctx} & FORCE_CATS

        resolved = forced_cats - cats_ctx      # 문맥 적용 후 사라진 강제 카테고리
        still_there = forced_cats & cats_ctx   # 여전히 남은 강제 카테고리

        title = r.get("title", "")
        print(f"[{short_name}] {title}")
        print(f"  원래 강제 카테고리: {sorted(forced_cats)}")
        print(f"  문맥 없이 재분류: {sorted(cats_no_ctx)}")
        print(f"  문맥 포함 재분류: {sorted(cats_ctx)}")
        if not still_there:
            print(f"  -> 해소됨 (강제 카테고리 전부 사라짐)")
            resolved_fp += 1
        elif resolved:
            print(f"  -> 일부 해소 (사라짐: {sorted(resolved)} / 남음: {sorted(still_there)})")
            partial_fp += 1
        else:
            print(f"  -> 해소 안 됨")
            unresolved_fp += 1
        print()

    total_tested = len(fp_forced) - load_failed
    print("=" * 60)
    print(f"  문맥 윈도우(size={args.context_window}) 효과 요약")
    print("=" * 60)
    print(f"  테스트: {total_tested}건 (파일 없어서 제외: {load_failed}건)")
    print(f"  완전 해소: {resolved_fp}건 ({resolved_fp/total_tested*100:.1f}%)" if total_tested else "")
    print(f"  일부 해소: {partial_fp}건")
    print(f"  해소 안 됨: {unresolved_fp}건")
    print("=" * 60)
    print("\n완전+일부 해소 비율이 높으면 문맥 윈도우가 유효하다는 뜻 -> ")
    print("evaluate_story()의 use_context=True를 프로덕션 기본값으로 바꾸는 것을 고려.")
    print("해소 안 된 사례가 많으면 재학습(옵션2 데이터 확장)이 더 필요하다는 뜻.")


if __name__ == "__main__":
    main()
