"""
test_safeguard_context.py
문맥 윈도우 방식(safeguard.py의 use_context=True)이 실제로 도메인 오탐(FP)을
줄이는지, 그리고 그 대가로 진짜 위험한 문장(정탐, TP) 탐지력까지 깎아먹지는
않는지 확인하는 A/B 테스트. 재학습 없이 프롬프트만 바꿔서 즉시 검증 가능.

--mode fp (기본): decision_source=kanana_force AND ground_truth=safe
      (사람은 안전하다고 봤는데 카나나가 S3/S5/S6로 강제 태깅한 오탐 사례)
      문맥을 넣었을 때 오탐 카테고리가 사라지는지 확인 (사라지면 좋음).
      단, 사라진 대신 다른 강제 카테고리가 새로 뜨면 "해소"가 아니라 "카테고리 이동"으로 별도 집계.

--mode tp: decision_source=kanana_force AND ground_truth=unsafe
      (진짜 위험해서 카나나가 정확히 강제 태깅한 정탐 사례)
      문맥을 넣었을 때도 여전히 잡히는지 확인 (여전히 잡혀야 좋음 — 안 잡히면 재현율 손실 = 나쁨).

GPU 필요 (pod에서 실행). 두 모드 다 돌려봐야 문맥 윈도우를 프로덕션에 적용해도
안전한지(오탐은 줄고 정탐은 안 깎이는지) 판단할 수 있다.

사용법:
    python test_safeguard_context.py --eval-jsonl kanana_solar_eval.jsonl --corpus-dir all_data --mode fp
    python test_safeguard_context.py --eval-jsonl kanana_solar_eval.jsonl --corpus-dir all_data --mode tp
    python test_safeguard_context.py --eval-jsonl kanana_solar_eval.jsonl --corpus-dir all_data --mode tp --limit 20
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


FORCE_CATS = {"S3", "S5", "S6"}


def run_fp_mode(safeguard, story_to_text, recs, corpus_dir, context_window, limit):
    """오탐(FP) 사례: 문맥을 넣었을 때 강제 카테고리가 사라지는지 확인."""
    targets = [
        r for r in recs
        if r.get("decision_source") == "kanana_force" and r.get("ground_truth") == "safe"
    ]
    if limit:
        targets = targets[:limit]
    print(f"[FP 모드] 테스트 대상 오탐 사례: {len(targets)}건\n")

    resolved = partial = shifted = unresolved = load_failed = 0

    for r in targets:
        short_name = r.get("file") or r.get("filename")
        forced_cats = set(r.get("forced_categories") or [])
        fname = resolve_filename(short_name, corpus_dir)
        path = os.path.join(corpus_dir, fname)
        if not os.path.exists(path):
            print(f"[건너뜀] {short_name} — 파일 없음")
            load_failed += 1
            continue

        with open(path, encoding="utf-8") as f:
            text = story_to_text(json.load(f))

        _, flagged_no_ctx = safeguard.evaluate_story(text, use_context=False)
        _, flagged_ctx = safeguard.evaluate_story(
            text, use_context=True, context_window=context_window
        )

        cats_no_ctx = {f["category"] for f in flagged_no_ctx} & FORCE_CATS
        cats_ctx = {f["category"] for f in flagged_ctx} & FORCE_CATS

        cleared = forced_cats - cats_ctx        # 원래 있던 강제 카테고리 중 사라진 것
        still_there = forced_cats & cats_ctx    # 원래 있던 강제 카테고리 중 남은 것
        new_cats = cats_ctx - forced_cats       # 문맥 적용 후 새로 뜬 강제 카테고리 (없던 게 생김 — 나쁜 신호)

        title = r.get("title", "")
        print(f"[{short_name}] {title}")
        print(f"  원래 강제 카테고리: {sorted(forced_cats)}")
        print(f"  문맥 없이 재분류: {sorted(cats_no_ctx)}")
        print(f"  문맥 포함 재분류: {sorted(cats_ctx)}")

        if not still_there and not new_cats:
            print(f"  -> 완전 해소")
            resolved += 1
        elif not still_there and new_cats:
            print(f"  -> 원래 카테고리는 사라졌지만 다른 강제 카테고리({sorted(new_cats)})가 새로 뜸 — 해소로 보지 않음")
            shifted += 1
        elif cleared:
            print(f"  -> 일부 해소 (사라짐: {sorted(cleared)} / 남음: {sorted(still_there)})")
            partial += 1
        else:
            print(f"  -> 해소 안 됨")
            unresolved += 1
        print()

    total = len(targets) - load_failed
    print("=" * 60)
    print(f"  [FP 모드] 문맥 윈도우 오탐 해소 효과")
    print("=" * 60)
    print(f"  테스트: {total}건 (파일 없어서 제외: {load_failed}건)")
    if total:
        print(f"  완전 해소: {resolved}건 ({resolved/total*100:.1f}%)")
        print(f"  카테고리 이동(해소 아님): {shifted}건 ({shifted/total*100:.1f}%)")
        print(f"  일부 해소: {partial}건")
        print(f"  해소 안 됨: {unresolved}건")
    print("=" * 60)


def run_tp_mode(safeguard, story_to_text, recs, corpus_dir, context_window, limit):
    """정탐(TP) 사례: 문맥을 넣어도 진짜 위험한 카테고리가 여전히 잡히는지 확인 (재현율 손실 점검)."""
    targets = [
        r for r in recs
        if r.get("decision_source") == "kanana_force" and r.get("ground_truth") == "unsafe"
    ]
    if limit:
        targets = targets[:limit]
    print(f"[TP 모드] 테스트 대상 정탐 사례: {len(targets)}건\n")

    preserved = lost = load_failed = 0

    for r in targets:
        short_name = r.get("file") or r.get("filename")
        forced_cats = set(r.get("forced_categories") or [])
        fname = resolve_filename(short_name, corpus_dir)
        path = os.path.join(corpus_dir, fname)
        if not os.path.exists(path):
            print(f"[건너뜀] {short_name} — 파일 없음")
            load_failed += 1
            continue

        with open(path, encoding="utf-8") as f:
            text = story_to_text(json.load(f))

        _, flagged_ctx = safeguard.evaluate_story(
            text, use_context=True, context_window=context_window
        )
        cats_ctx = {f["category"] for f in flagged_ctx} & FORCE_CATS
        still_caught = forced_cats & cats_ctx
        lost_cats = forced_cats - cats_ctx

        title = r.get("title", "")
        print(f"[{short_name}] {title}")
        print(f"  원래 강제 카테고리(정탐): {sorted(forced_cats)}")
        print(f"  문맥 포함 재분류: {sorted(cats_ctx)}")
        if not lost_cats:
            print(f"  -> 유지됨 (여전히 잡힘)")
            preserved += 1
        else:
            print(f"  -> ⚠ 놓침 ({sorted(lost_cats)}이 문맥 적용 후 안 잡힘 — 재현율 손실 위험)")
            lost += 1
        print()

    total = len(targets) - load_failed
    print("=" * 60)
    print(f"  [TP 모드] 문맥 윈도우가 정탐 탐지력에 미치는 영향")
    print("=" * 60)
    print(f"  테스트: {total}건 (파일 없어서 제외: {load_failed}건)")
    if total:
        print(f"  유지됨: {preserved}건 ({preserved/total*100:.1f}%)")
        print(f"  놓침(재현율 손실): {lost}건 ({lost/total*100:.1f}%)")
    print("=" * 60)
    if total and lost / total > 0.1:
        print("\n⚠ 놓친 비율이 10% 넘음 — 문맥 윈도우를 그냥 기본값으로 켜면 안전 탐지력이 눈에 띄게 깎일 수 있음.")
    else:
        print("\n놓친 비율이 낮으면 문맥 윈도우를 프로덕션 기본값으로 켜도 안전 탐지력 손실이 크지 않다는 뜻.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--corpus-dir", default="all_data")
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N건만 테스트")
    ap.add_argument("--context-window", type=int, default=1)
    ap.add_argument("--mode", choices=["fp", "tp"], default="fp",
                     help="fp: 오탐 해소 효과 확인(기본) / tp: 정탐 유지(재현율 손실) 확인")
    args = ap.parse_args()

    from src.data_loader import story_to_text
    from src.safeguard import KananaSafeguard

    recs = []
    with open(args.eval_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    print("세이프가드 로딩...")
    safeguard = KananaSafeguard()

    if args.mode == "fp":
        run_fp_mode(safeguard, story_to_text, recs, args.corpus_dir, args.context_window, args.limit)
    else:
        run_tp_mode(safeguard, story_to_text, recs, args.corpus_dir, args.context_window, args.limit)


if __name__ == "__main__":
    main()
