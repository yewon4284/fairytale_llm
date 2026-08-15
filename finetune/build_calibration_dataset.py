"""
finetune/build_calibration_dataset.py
세이프가드 도메인 오탐(FP) 개선용 LoRA 재보정 데이터셋 생성.

kanana_solar_eval.jsonl(강제unsafe 판정 근거 파일)을 읽어서:
  - FP(사람=safe, 카나나=강제unsafe로 오탐)로 걸린 스토리
    -> 현재 세이프가드를 다시 돌려서 실제로 어떤 문장이 걸렸는지 재현
    -> 그 문장을 SAFE로 재라벨 (hard negative — "이건 오탐이었다"를 가르치는 예시)
  - TP(사람=unsafe, 카나나=강제unsafe로 정확히 잡음)로 걸린 스토리
    -> 걸린 문장을 UNSAFE-Sx 그대로 유지 (positive — 진짜 위반 탐지력을 깎아먹지 않기 위함)
를 뽑아서 finetune/train_s5.py와 동일한 스키마([{"text":..., "label":...}])로 저장한다.
train_s5.py는 label 문자열을 그대로 <label> 완성 토큰으로 쓰므로 수정 없이 그대로 재사용 가능
(SAFE/UNSAFE-S3/UNSAFE-S5/UNSAFE-S6 다 지원됨).

GPU 필요 (세이프가드 모델 로딩, pod에서 실행).

사용법:
    python finetune/build_calibration_dataset.py \
        --eval-jsonl kanana_solar_eval.jsonl \
        --corpus-dir all_data \
        --output finetune/calibration_dataset.json \
        --merge-with finetune/s5_dataset.json
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", required=True,
                     help="kanana_solar_eval.jsonl 형식 (file, ground_truth, decision_source, forced_categories 필드 필요)")
    ap.add_argument("--corpus-dir", default="all_data", help="원본 동화 JSON들이 있는 폴더")
    ap.add_argument("--output", default="finetune/calibration_dataset.json")
    ap.add_argument("--merge-with", default=None,
                     help="기존 데이터셋(예: finetune/s5_dataset.json)과 합칠 경우 경로. "
                          "텍스트가 겹치면 새로 뽑은 라벨을 우선함")
    ap.add_argument("--no-adapter", action="store_true",
                     help="현재 로딩된 S5 어댑터 없이(베이스 모델로) 재현 — 어댑터 자체가 오탐 원인인지 "
                          "따로 확인하고 싶을 때. 기본은 어댑터 켠 상태(=현재 프로덕션과 동일 조건)")
    args = ap.parse_args()

    from src.data_loader import story_to_text
    from src.safeguard import KananaSafeguard

    # kanana_solar_eval.jsonl의 "file" 필드(예: "01_17.json")는 실제 all_data 파일명이 아니라,
    # 카테고리 폴더(all_data 안에서 파일명 접두사 "03_01"~"03_05" 5종, 각 88/175/54/67/57개 = 441개)별로
    # 파일명을 정렬한 뒤 1-based 인덱스를 매긴 축약 표기임 (예: "01_17" = "03_01" 폴더에서 정렬 17번째).
    # 실제 발견 경위: 01_17.json(title="방귀 시합")이 all_data/03_01T_01S_9791157987085.json과 일치함을 확인.
    CATEGORY_PREFIX = {"01": "03_01", "02": "03_02", "03": "03_03", "04": "03_04", "05": "03_05"}
    _category_cache = {}

    def resolve_filename(short_name, corpus_dir):
        """'01_17.json' -> 실제 all_data 파일명으로 변환. 실패하면 short_name 그대로 반환(원래도 직접
        일치하는 케이스 대비)."""
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

    recs = []
    with open(args.eval_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    forced = [r for r in recs if r.get("decision_source") == "kanana_force"]
    fp_forced = [r for r in forced if r.get("ground_truth") == "safe"]
    tp_forced = [r for r in forced if r.get("ground_truth") == "unsafe"]
    print(f"강제unsafe 총 {len(forced)}건 — FP(오탐, SAFE로 재라벨 대상) {len(fp_forced)}건 "
          f"/ TP(정탐, 원라벨 유지) {len(tp_forced)}건")

    print(f"세이프가드 로딩 (S5 어댑터 {'끔' if args.no_adapter else '켬 — 현재 프로덕션과 동일 조건'})...")
    safeguard = KananaSafeguard(use_s5_adapter=not args.no_adapter)

    examples = []

    def process(records, relabel_safe):
        for r in records:
            fname = r.get("file") or r.get("filename")
            if not fname:
                continue
            path = os.path.join(args.corpus_dir, fname)
            if not os.path.exists(path):
                print(f"  [건너뜀] {fname} — {args.corpus_dir}에 파일 없음")
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            text = story_to_text(data)
            if not text.strip():
                print(f"  [건너뜀] {fname} — 본문 없음")
                continue

            _, flagged = safeguard.evaluate_story(text)
            if not flagged:
                print(f"  [주의] {fname} — 지금 다시 돌리니 태깅된 문장이 0개(원래와 다름, "
                      f"모델/어댑터 상태가 바뀌었을 수 있음). 건너뜀.")
                continue

            for fl in flagged:
                label = "SAFE" if relabel_safe else f"UNSAFE-{fl['category']}"
                examples.append({
                    "text": fl["sentence"],
                    "label": label,
                    "source_file": fname,
                    "orig_category": fl["category"],
                })
            kind = "SAFE로 재라벨" if relabel_safe else "원 라벨 유지"
            print(f"  {fname}: 태깅 문장 {len(flagged)}개 -> {kind}")

    print("\n[1/2] FP 사례 처리 — 오탐 문장을 SAFE로 재라벨")
    process(fp_forced, relabel_safe=True)

    print("\n[2/2] TP 사례 처리 — 정탐 문장은 원 라벨 유지(탐지력 보존)")
    process(tp_forced, relabel_safe=False)

    if args.merge_with and os.path.exists(args.merge_with):
        with open(args.merge_with, encoding="utf-8") as f:
            existing = json.load(f)
        seen_texts = {e["text"] for e in examples}
        added = 0
        for e in existing:
            if e.get("text") not in seen_texts:
                examples.append({"text": e["text"], "label": e["label"]})
                seen_texts.add(e["text"])
                added += 1
        print(f"\n{args.merge_with}에서 겹치지 않는 {added}개 추가 병합")

    print(f"\n최종 {len(examples)}개 -> {args.output}")
    print("라벨 분포:", Counter(e["label"] for e in examples))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
