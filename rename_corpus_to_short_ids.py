"""
rename_corpus_to_short_ids.py
all_data(및 data_sorted 계열 하위 폴더들)의 원본 파일명(예: 03_01T_01S_9791157987085.json,
ISBN 등을 담은 긴 이름)을, kanana_solar_eval.jsonl 등 팀원 스크립트가 쓰는 축약 표기
("01_17.json" = all_data 내 접두사 "03_01" 그룹에서 정렬 17번째)로 통일해서 실제로 리네임한다.

배경: kanana_solar_eval.jsonl의 file 필드가 "01_17.json" 형태인데 all_data 실제 파일명과
전혀 안 맞아서, 지금까지는 build_calibration_dataset.py에서 런타임에 역산해 매칭했음.
이 스크립트로 파일명 자체를 그 축약 표기에 맞춰 바꿔두면 이후 모든 스크립트/팀원 산출물이
파일명 기준으로 그냥 매칭된다.

대상 폴더: all_data, data_sorted, data_sorted_cat_*, data_sorted_mixed
  (all_data 기준으로 카테고리/순번을 계산한 뒤, 같은 원본 파일명을 가진 다른 폴더의 파일도
   동일한 새 이름으로 리네임 — 폴더 간 참조 일관성 유지)

매핑 규칙: all_data 파일명 접두사 "03_01"~"03_05" (5개 카테고리) 별로 파일명을 정렬해
1-based 인덱스를 매기고, "{01~05}_{인덱스}.json"으로 리네임.

사용법:
    python rename_corpus_to_short_ids.py --dry-run   # 미리보기만, 실제 변경 없음
    python rename_corpus_to_short_ids.py              # 실제 리네임 수행
    (완료 후 finetune/corpus_rename_map.json에 원본명<->새이름 매핑 저장 — 감사/복구용)
"""

import argparse
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
ALL_DATA_DIR = os.path.join(ROOT, "all_data")
CATEGORY_PREFIXES = ["03_01", "03_02", "03_03", "03_04", "03_05"]
PREFIX_TO_SHORT_CAT = {"03_01": "01", "03_02": "02", "03_03": "03", "03_04": "04", "03_05": "05"}

OTHER_DIRS = [
    "data_sorted",
    "data_sorted_cat_사회관계",
    "data_sorted_cat_신체운동_건강",
    "data_sorted_cat_예술경험",
    "data_sorted_cat_의사소통",
    "data_sorted_cat_자연탐구",
    "data_sorted_mixed",
]


def build_mapping():
    """all_data 안의 원본 파일명 -> 새 축약 파일명 매핑을 만든다."""
    mapping = {}
    for prefix in CATEGORY_PREFIXES:
        files = sorted(f for f in os.listdir(ALL_DATA_DIR) if f.startswith(prefix))
        short_cat = PREFIX_TO_SHORT_CAT[prefix]
        for i, fname in enumerate(files, 1):
            mapping[fname] = f"{short_cat}_{i:02d}.json"
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="실제 리네임 없이 매핑만 미리 보기")
    ap.add_argument("--map-output", default=os.path.join(ROOT, "finetune", "corpus_rename_map.json"))
    args = ap.parse_args()

    mapping = build_mapping()
    print(f"all_data {len(mapping)}개 파일에 대한 매핑 생성 완료")
    print("샘플 5개:")
    for i, (old, new) in enumerate(mapping.items()):
        if i >= 5:
            break
        print(f"  {old}  ->  {new}")

    # 새 이름끼리 충돌 없는지 확인
    new_names = list(mapping.values())
    if len(set(new_names)) != len(new_names):
        print("[오류] 새 이름에 중복이 있습니다. 중단합니다.")
        return

    # 대상 폴더 목록: all_data + 존재하는 다른 폴더들
    targets = [ALL_DATA_DIR]
    for d in OTHER_DIRS:
        path = os.path.join(ROOT, d)
        if os.path.isdir(path):
            targets.append(path)
        else:
            print(f"  (참고: {d} 폴더 없음, 건너뜀)")

    if args.dry_run:
        print("\n[dry-run] 실제 변경 없음. 아래 폴더들에 리네임이 적용될 예정입니다:")
        for t in targets:
            n = sum(1 for f in os.listdir(t) if f in mapping)
            print(f"  {t}: {n}개 파일 리네임 대상")
        return

    total_renamed = 0
    for target_dir in targets:
        renamed_here = 0
        for old_name, new_name in mapping.items():
            old_path = os.path.join(target_dir, old_name)
            if not os.path.exists(old_path):
                continue
            new_path = os.path.join(target_dir, new_name)
            if os.path.exists(new_path):
                print(f"  [건너뜀] {target_dir}: {new_name} 이미 존재함 (덮어쓰지 않음)")
                continue
            os.rename(old_path, new_path)
            renamed_here += 1
        print(f"{target_dir}: {renamed_here}개 리네임 완료")
        total_renamed += renamed_here

    os.makedirs(os.path.dirname(args.map_output), exist_ok=True)
    with open(args.map_output, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"\n총 {total_renamed}개 파일 리네임 완료 (전체 대상 폴더 합산)")
    print(f"원본명<->새이름 매핑 저장: {args.map_output}")


if __name__ == "__main__":
    main()
