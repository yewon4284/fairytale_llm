"""
data_loader.py
───────────────
아동 동화 데이터셋(JSON) 로더.
'의사소통' 분류 동화만 필터링하여 Few-shot 예시 등으로 활용.
"""

import json
import os
import glob
import random
from typing import List, Dict, Optional


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_all_stories(data_dir: str = DATA_DIR) -> List[Dict]:
    """data/ 폴더 안의 JSON 파일을 모두 로드."""
    stories = []
    pattern = os.path.join(data_dir, "*.json")
    for filepath in glob.glob(pattern):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            stories.append(data)
        except Exception as e:
            print(f"[DataLoader] {filepath} 로드 실패: {e}")
    return stories


def filter_by_classification(
    stories: List[Dict],
    classification: str = "의사소통"
) -> List[Dict]:
    """특정 분류(classification)에 해당하는 동화만 반환."""
    return [s for s in stories if s.get("classification") == classification]


def story_to_text(story: Dict) -> str:
    """
    동화 JSON을 srcPage 순서대로 연결한 전체 텍스트로 변환.
    paragraphInfo가 없으면 빈 문자열 반환.
    """
    paragraphs = story.get("paragraphInfo", [])
    if not paragraphs:
        return ""
    # srcPage 기준 정렬
    paragraphs_sorted = sorted(paragraphs, key=lambda p: p.get("srcPage", 0))
    return "\n".join(p.get("srcText", "") for p in paragraphs_sorted)


def get_reference_stories(
    data_dir: str = DATA_DIR,
    classification: str = "의사소통",
    n: int = 3,
    seed: Optional[int] = 42,
) -> List[Dict]:
    """
    Few-shot 프롬프트용 참고 동화 n편을 무작위 추출.
    반환 형식: [{"title": ..., "text": ...}, ...]
    """
    all_stories = load_all_stories(data_dir)
    filtered = filter_by_classification(all_stories, classification)
    if seed is not None:
        random.seed(seed)
    sample = random.sample(filtered, min(n, len(filtered)))
    return [
        {
            "title": s.get("title", "제목 없음"),
            "readAge": s.get("readAge", "유아"),
            "text": story_to_text(s),
        }
        for s in sample
    ]


def get_dataset_stats(data_dir: str = DATA_DIR) -> Dict:
    """데이터셋 통계 출력용."""
    all_stories = load_all_stories(data_dir)
    classifications = {}
    for s in all_stories:
        cls = s.get("classification", "미분류")
        classifications[cls] = classifications.get(cls, 0) + 1
    return {
        "total": len(all_stories),
        "by_classification": classifications,
    }


if __name__ == "__main__":
    stats = get_dataset_stats()
    print(f"전체 동화 수: {stats['total']}")
    print("분류별:")
    for k, v in sorted(stats["by_classification"].items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}편")

    refs = get_reference_stories(n=2)
    for r in refs:
        print(f"\n[참고 동화] {r['title']}")
        print(r["text"][:200], "...")