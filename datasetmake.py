import json
import os
import glob

# 경로 설정
DATA_DIR = "/home/user2/fairytale_llm/data_sorted"
OUTPUT_PATH = "/home/user2/fairytale_llm/test_data/testset_SJ.json"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

def build_story_text(data: dict) -> str:
    return "\n".join(p["srcText"] for p in data["paragraphInfo"])

def get_type(idx: int) -> str:
    if idx <= 20:   return "A"
    elif idx <= 40: return "B"
    elif idx <= 60: return "C"
    else:           return "D"

def get_ground_truth(type_: str) -> str:
    return "SAFE" if type_ == "D" else "UNSAFE"

files = sorted(glob.glob(os.path.join(DATA_DIR, "03_01T_01S_*.json")))
print(f"총 {len(files)}개 파일 발견\n")

results = []
for idx, filepath in enumerate(files, start=1):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    type_ = get_type(idx)

    entry = {
        "id": f"{type_}-{idx:03d}",
        "type": type_,
        "ground_truth": get_ground_truth(type_),
        "description": "",
        "inserted_sentence": None if type_ == "D" else "",
        "story_title": data.get("title", ""),
        "story_text": build_story_text(data),
        "source_file": os.path.basename(filepath)
    }
    results.append(entry)
    print(f"[{entry['id']}] {entry['story_title']}")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n완료: {len(results)}개 → {OUTPUT_PATH}")