"""
finetune/mine_domain_safe_examples.py
세이프가드 도메인 오탐(FP) 재보정용 SAFE 예시를 폭넓게 확충한다.

build_calibration_dataset.py는 실제 FP로 걸렸던 44건 문장만 SAFE로 재라벨하는데,
이러면 모델이 "이 정확한 문장들만 안전하다"를 암기해버리고 일반화가 안 될 위험이 있다
(실측: 하이퍼파라미터를 고쳐도 성능이 크게 안 나아졌다는 팀원 피드백).

이 스크립트는 알려진 4가지 도메인 오탐 패턴(곤충/과학, 배변/위생, 음식안전, 신체경계)의
핵심 어휘가 들어간 문장을, 사람이 실제로 safe라고 라벨링한 스토리들(kanana_solar_eval.jsonl
기준) 전체에서 찾아 SAFE 예시로 추가한다. FP로 걸렸던 문장에 국한하지 않고 "이런 종류의
단어/맥락이면 대체로 안전하다"는 더 일반적인 경계를 학습시키는 것이 목적.

주의: SAFE 예시만 잔뜩 늘리면 반대로 진짜 위험한 문장까지 안전하다고 오판하는 쪽(재현율
하락)으로 치우칠 수 있다. build_calibration_dataset.py가 만드는 TP(정탐) 예시 + 기존
s5_dataset.json의 UNSAFE 예시와 반드시 같이 합쳐서 균형을 확인할 것 — 이 스크립트 단독
출력은 SAFE 쪽으로 편향돼 있다.

사용법:
    python finetune/mine_domain_safe_examples.py \
        --eval-jsonl kanana_solar_eval.jsonl \
        --corpus-dir all_data \
        --output finetune/domain_safe_examples.json \
        --per-group-limit 40

    # build_calibration_dataset.py 산출물과 바로 합치려면:
    python finetune/mine_domain_safe_examples.py \
        --eval-jsonl kanana_solar_eval.jsonl --corpus-dir all_data \
        --output finetune/calibration_dataset.json \
        --merge-with finetune/calibration_dataset.json
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 4가지 알려진 도메인 오탐 패턴별 핵심 어휘 (evaluator.py의 재검토 규칙 코멘트와 동일 근거)
KEYWORD_GROUPS = {
    "곤충_과학": ["곤충", "벌레", "거미", "나비", "개미", "잠자리", "애벌레", "버섯",
                "지렁이", "딱정벌레", "매미", "달팽이", "번데기"],
    "배변_위생": ["응가", "똥", "오줌", "방귀", "화장실", "변기", "배변", "기저귀", "목욕"],
    "음식안전": ["독버섯", "독이 있", "독을 품", "상한 음식", "상한 냄새", "유통기한",
               "식중독", "함부로 먹", "안전하게 먹", "썩은 음식", "곰팡이가 핀"],
    "신체경계": ["엉덩이", "뽀뽀", "만지", "안아", "신체", "비밀", "사생활"],
}

MIN_SENTENCE_LEN = 8  # 너무 짧은 문장(노이즈) 제외


def resolve_filename(short_name, corpus_dir, category_cache):
    direct = os.path.join(corpus_dir, short_name)
    if os.path.exists(direct):
        return short_name
    base = short_name[:-5] if short_name.endswith(".json") else short_name
    parts = base.split("_")
    prefix_map = {"01": "03_01", "02": "03_02", "03": "03_03", "04": "03_04", "05": "03_05"}
    if len(parts) != 2 or parts[0] not in prefix_map:
        return short_name
    cat, idx_str = parts
    try:
        idx = int(idx_str)
    except ValueError:
        return short_name
    prefix = prefix_map[cat]
    if prefix not in category_cache:
        category_cache[prefix] = sorted(
            fn for fn in os.listdir(corpus_dir) if fn.startswith(prefix)
        )
    files = category_cache[prefix]
    if 1 <= idx <= len(files):
        return files[idx - 1]
    return short_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--corpus-dir", default="all_data")
    ap.add_argument("--output", default="finetune/domain_safe_examples.json")
    ap.add_argument("--merge-with", default=None,
                     help="기존 데이터셋과 합쳐서 --output에 저장 (텍스트 중복 제거)")
    ap.add_argument("--per-group-limit", type=int, default=40,
                     help="어휘 그룹별 최대 채택 문장 수 (한 스토리에 쏠리지 않게 스토리별 1문장 우선)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-files", default=None,
                     help="split_train_holdout.py가 만든 holdout_files.json 경로. 지정하면 이 목록에 "
                          "있는 스토리는 마이닝 대상에서 제외 (진짜 held-out 검증을 위해 필수)")
    args = ap.parse_args()

    from src.data_loader import story_to_text
    from src.safeguard import SENTENCE_SPLIT_RE

    random.seed(args.seed)

    excluded = set()
    if args.exclude_files:
        with open(args.exclude_files, encoding="utf-8") as f:
            excluded = set(json.load(f))
        print(f"held-out 제외 목록 로딩: {len(excluded)}건")

    recs = []
    with open(args.eval_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    if excluded:
        before_n = len(recs)
        recs = [r for r in recs if (r.get("file") or r.get("filename")) not in excluded]
        print(f"held-out 제외 적용: {before_n}건 -> {len(recs)}건")

    safe_recs = [r for r in recs if r.get("ground_truth") == "safe"]
    print(f"ground_truth=safe 스토리 {len(safe_recs)}건에서 도메인 어휘 문장 탐색")

    category_cache = {}
    # group -> list of (sentence, source_file)
    candidates = defaultdict(list)

    for r in safe_recs:
        short_name = r.get("file") or r.get("filename")
        if not short_name:
            continue
        fname = resolve_filename(short_name, args.corpus_dir, category_cache)
        path = os.path.join(args.corpus_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        text = story_to_text(data)
        if not text.strip():
            continue

        sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
        # 스토리 하나에서 그룹당 최대 1문장만 채택 (한 스토리에 쏠리는 것 방지)
        matched_this_story = set()
        for sent in sentences:
            if len(sent) < MIN_SENTENCE_LEN:
                continue
            for group, keywords in KEYWORD_GROUPS.items():
                if group in matched_this_story:
                    continue
                if any(kw in sent for kw in keywords):
                    candidates[group].append((sent, short_name))
                    matched_this_story.add(group)

    examples = []
    seen_texts = set()
    for group, items in candidates.items():
        random.shuffle(items)
        taken = 0
        for sent, source in items:
            if taken >= args.per_group_limit:
                break
            if sent in seen_texts:
                continue
            seen_texts.add(sent)
            examples.append({
                "text": sent,
                "label": "SAFE",
                "source_file": source,
                "domain_group": group,
            })
            taken += 1
        print(f"  {group}: 후보 {len(items)}건 중 {taken}건 채택")

    if args.merge_with and os.path.exists(args.merge_with):
        # merge_with와 output이 같은 경로여도 된다 — 여기서 먼저 읽어서 메모리에 올려두고,
        # 실제 쓰기는 이 아래에서 별도로 하기 때문에 self-merge(같은 파일에 이어붙이기)가 안전하게 동작함.
        with open(args.merge_with, encoding="utf-8") as f:
            existing = json.load(f)
        added = 0
        for e in existing:
            if e.get("text") not in seen_texts:
                examples.append({"text": e["text"], "label": e["label"]})
                seen_texts.add(e["text"])
                added += 1
        print(f"{args.merge_with}에서 겹치지 않는 {added}개 추가 병합")

    print(f"\n최종 {len(examples)}개 -> {args.output}")
    print("라벨 분포:", Counter(e["label"] for e in examples))
    print("\n⚠ 이 스크립트 단독 출력은 SAFE 쪽으로 편향돼 있습니다. build_calibration_dataset.py의 "
          "TP 예시 및 s5_dataset.json의 UNSAFE 예시와 반드시 같이 합쳐서 라벨 균형을 확인하세요.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
