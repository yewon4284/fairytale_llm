"""
data_loader.py
data/ 폴더의 동화 JSON 데이터셋을 로드합니다.

활용 방향:
- few-shot 생성 가이드: "의사소통" 분류 동화 → 대화 중심, 동화체 문체 가이드
- 평가 캘리브레이션: Solar 평가 시 "이 정도가 기준점" 제시용
"""

import json
import random
from pathlib import Path
from typing import Optional


def load_fairy_tales(data_dir: str = "data") -> list[dict]:
    """data/ 폴더의 모든 JSON 동화 파일을 로드합니다."""
    tales = []
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"[DataLoader] 경고: {data_dir} 폴더가 없습니다.")
        return tales
    for json_file in data_path.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                tales.append(json.load(f))
        except Exception as e:
            print(f"[DataLoader] {json_file.name} 로드 실패: {e}")
    print(f"[DataLoader] {len(tales)}개 동화 로드 완료")
    return tales


def extract_full_text(tale: dict) -> str:
    """srcPage 기준으로 정렬한 전체 텍스트를 반환합니다."""
    paragraphs = sorted(
        tale.get("paragraphInfo", []), key=lambda p: p.get("srcPage", 0)
    )
    return "\n".join(p["srcText"] for p in paragraphs)


def get_few_shot_samples(
    tales: list[dict],
    n: int = 3,
    seed: int = 42,
) -> list[dict]:
    """
    Generator few-shot 가이드용 샘플을 반환합니다.

    "의사소통" 분류 동화를 우선 선택합니다.
    이 분류는 대화 중심, 짧은 문장, 동화체 문체가 특징입니다.

    Returns:
        [{"title": str, "text": str, "word_count": int}, ...]
    """
    comm_tales = [t for t in tales if t.get("classification") == "의사소통"]
    pool = comm_tales if len(comm_tales) >= n else tales
    print(f"[DataLoader] few-shot 풀: {len(pool)}개 (의사소통 {len(comm_tales)}개)")

    random.seed(seed)
    sampled = random.sample(pool, min(n, len(pool)))
    result = []
    for tale in sampled:
        text = extract_full_text(tale)
        word_count = sum(
            p.get("srcWordEA", 0) for p in tale.get("paragraphInfo", [])
        )
        result.append({
            "title": tale.get("title", "제목 없음"),
            "text": text,
            "word_count": word_count,
        })
    return result


def build_few_shot_block(samples: list[dict], chars_per_sample: int = 300) -> str:
    """
    Generator 시스템 프롬프트에 삽입할 few-shot 블록을 만듭니다.
    각 샘플은 앞부분 chars_per_sample 글자만 사용합니다.
    """
    if not samples:
        return ""
    lines = ["[참고 동화 예시 - 문체와 리듬감을 참고하세요]"]
    for i, s in enumerate(samples, 1):
        preview = s["text"][:chars_per_sample]
        if len(s["text"]) > chars_per_sample:
            preview += "..."
        lines.append(f"\n--- 예시 {i}: {s['title']} ({s['word_count']}단어) ---")
        lines.append(preview)
    return "\n".join(lines)


def get_calibration_sample(tales: list[dict], seed: int = 42) -> Optional[dict]:
    """
    Solar 평가 캘리브레이션용 샘플 1개를 반환합니다.
    의사소통 분류에서 선택합니다.
    """
    comm_tales = [t for t in tales if t.get("classification") == "의사소통"]
    pool = comm_tales if comm_tales else tales
    if not pool:
        return None
    random.seed(seed)
    tale = random.choice(pool)
    text = extract_full_text(tale)
    word_count = sum(p.get("srcWordEA", 0) for p in tale.get("paragraphInfo", []))
    return {
        "title": tale.get("title", "제목 없음"),
        "text": text,
        "word_count": word_count,
    }