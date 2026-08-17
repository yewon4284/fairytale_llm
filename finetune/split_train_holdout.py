"""
finetune/split_train_holdout.py
kanana_solar_eval.jsonl(421편, ground_truth 라벨 포함)을 재보정 데이터 생성용
"train-source"(80%)와 절대 학습에 안 쓸 "held-out"(20%)으로 미리 쪼갠다.

왜 필요한가:
  build_calibration_dataset.py / mine_domain_safe_examples.py가 재보정 학습 데이터를
  만들 때 바로 이 421편에서 문장을 뽑아왔음. 그래서 지금까지의 성능 검증(FP 44건 중
  17건 해소 등)은 학습에 쓴 것과 같은 스토리 풀로 평가한 것이라 순환 검증 우려가 있음
  (2026-08-16 진단). 이 스크립트로 미리 20%를 떼어놓고 그 부분은 데이터 생성 단계부터
  절대 건드리지 않으면, 그 20%로 평가한 결과는 진짜 held-out 성능이 됨.

층화 방식: (decision_source, ground_truth) 조합별로 80/20 분할해서 각 부분집합의
비율이 원본과 비슷하게 유지되도록 함 (kanana_force+safe, kanana_force+unsafe,
solar+safe, solar+unsafe 4개 층).

사용법:
    python finetune/split_train_holdout.py \
        --eval-jsonl kanana_solar_eval.jsonl \
        --output-dir finetune/ \
        --holdout-ratio 0.2
"""

import argparse
import json
import os
import random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", default="kanana_solar_eval.jsonl")
    ap.add_argument("--output-dir", default="finetune")
    ap.add_argument("--holdout-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    recs = []
    with open(args.eval_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    strata = defaultdict(list)
    for r in recs:
        fname = r.get("file") or r.get("filename")
        if not fname:
            continue
        key = (r.get("decision_source"), r.get("ground_truth"))
        strata[key].append(fname)

    train_files = []
    holdout_files = []
    print(f"전체 {len(recs)}편, 층화 {args.holdout_ratio*100:.0f}% held-out 분할 (seed={args.seed})\n")
    for key, files in sorted(strata.items(), key=lambda kv: str(kv[0])):
        files = files[:]  # copy
        random.shuffle(files)
        n_holdout = max(1, round(len(files) * args.holdout_ratio)) if len(files) >= 5 else 0
        holdout = files[:n_holdout]
        train = files[n_holdout:]
        train_files.extend(train)
        holdout_files.extend(holdout)
        print(f"  {key}: 총 {len(files)}건 -> train {len(train)} / holdout {len(holdout)}")

    print(f"\n총 train-source {len(train_files)}건 / held-out {len(holdout_files)}건")
    print("⚠ held-out 파일들은 앞으로 build_calibration_dataset.py / mine_domain_safe_examples.py를 "
          "돌릴 때 반드시 --exclude-files로 제외해야 함. 안 그러면 이 분할이 무의미해짐.")

    os.makedirs(args.output_dir, exist_ok=True)
    holdout_path = os.path.join(args.output_dir, "holdout_files.json")
    train_path = os.path.join(args.output_dir, "train_source_files.json")
    with open(holdout_path, "w", encoding="utf-8") as f:
        json.dump(sorted(holdout_files), f, ensure_ascii=False, indent=2)
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(sorted(train_files), f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {holdout_path}, {train_path}")


if __name__ == "__main__":
    main()
