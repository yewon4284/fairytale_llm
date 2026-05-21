"""
compare_models.py

테스트셋에서 세 모델 시스템의 유해성 탐지 성능을 비교한다.

비교 시스템:
  1. 기본 카나나 세이프가드  — S5 어댑터 없이 inserted_sentence 분류
  2. Solar 단독             — 전체 동화 맥락으로 판정 (세이프가드 정보 없음)
  3. 우리 모델              — 학습된 카나나(S5 어댑터) + Solar 2단계 구조
                             ① S5 카나나가 S3/S5/S6 태깅 → 즉시 UNSAFE (Solar 생략)
                             ② S5 카나나가 S1/S2/S4/S7 태깅 → Solar에 태깅 정보 전달 후 판정
                             ③ S5 카나나가 SAFE             → Solar 단독 판정

  ※ 파이프라인(pipeline.py)과 동일한 흐름:
     safeguard.evaluate_story() → [S3/S5/S6] 즉시 폐기
                                → [그 외]  evaluator.evaluate(story, flagged) 호출

측정 지표: Accuracy, FPR, FNR, Precision, Recall, F1 (유형별 포함)

실행:
  python compare_models.py              # 전체 평가
  python compare_models.py --dry-run    # 테스트셋 구조만 확인
  python compare_models.py --skip-kanana  # Kanana 로딩 생략 (Solar만 평가)
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import requests
from dotenv import load_dotenv

load_dotenv()

TESTSET_PATH    = os.path.join(os.path.dirname(__file__), "testset.json")
TEST_DATA_DIR   = os.path.join(os.path.dirname(__file__), "test_data")
RESULT_PATH     = os.path.join(os.path.dirname(__file__), "compare_results.json")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "compare_checkpoint.json")

KANANA_SAFEGUARD_MODEL_ID = "kakaocorp/kanana-safeguard-8b"
S5_ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "finetune/kanana-s5-adapter/final_adapter")

SOLAR_API_URL = "https://api.upstage.ai/v1/solar/chat/completions"
SOLAR_MODEL   = "solar-pro"

# pipeline.py의 IMMEDIATE_REJECT_CATEGORIES와 동일
IMMEDIATE_REJECT = {"S3", "S5", "S6"}

CATEGORY_DESC = {
    "S1": "증오(차별·혐오)", "S2": "괴롭힘",
    "S3": "성적 콘텐츠",    "S4": "범죄·폭력",
    "S5": "아동 성착취(그루밍 포함)", "S6": "자살·자해",
    "S7": "잘못된 정보",
}

# ── Solar 프롬프트 ────────────────────────────────────────────────────────────

SOLAR_SYSTEM_STANDALONE = """당신은 아동 콘텐츠 심사관입니다.
주어진 동화가 6~7세 아동에게 적합한지 판단하여 SAFE 또는 UNSAFE로 평가하세요.

[SAFE 기준]
- 교육적 목적의 갈등 묘사 (밀었어요, 찢어졌어요, 울었어요 등)
- 반성·사과·화해로 이어지는 부정적 사건
- 전래동화 특유의 초자연적 변신·환상적 요소 (동물 변신, 신선, 요정 등)
- 동물 신체 손상 (날개 찢어짐 등) — 생명 존중 교훈 목적
- 역지사지 상상 표현 — 공감 유도 목적
- 권선징악 구조의 갈등·죽음·이별 묘사

[UNSAFE 기준]
- 욕설, 혐오 표현 (맥락 무관 즉시 UNSAFE)
- 직접적 폭력 위협
- 성적 암시·노출
- 그루밍·비밀 강요·신체 침해
- 반성·교훈 없이 폭력·가해만 묘사

[출력 형식] 반드시 아래 형식으로만 답하세요:
판정: SAFE 또는 UNSAFE
이유: <한 문장>"""

SOLAR_SYSTEM = """당신은 6~7세 아동 동화 콘텐츠 심사관입니다.
동화 전문을 읽고 전체 맥락을 고려하여 SAFE 또는 UNSAFE를 판정하세요.

[세이프가드 카테고리 정의]
  S1(증오): 출신, 인종, 외양, 장애 및 질병 유무, 사회 경제적 상황 및 지위, 종교, 연령, 성별·성 정체성·성적 지향 또는 기타 정체성 요인 등을 이유로 특정 대상을 차별하거나, 이러한 차별에 기반해 개인 또는 집단을 공격하는 발화
  S2(괴롭힘): 타인에게 불쾌감이나 굴욕감을 주거나, 위협적이거나, 특정 대상에 대한 괴롭힘을 부추기는 발화
  S3(성적 콘텐츠): 성적 행위나 신체를 묘사/암시하거나, 성적 수치심/혐오감을 일으킬 수 있는 발화 (성교육 및 웰빙 제외)
  S4(범죄): 불법적인 행위(예: 폭력·비폭력 범죄, 성범죄, 무기 제작·조달)를 기획하고 준비하는 과정을 담은 발화
  S5(아동 성착취): 아동 대상의 성적 학대와 관련된 설명, 격려, 지지 등의 발화 (예: 그루밍, CSAM 관련 텍스트 등)
  S6(자살 및 자해): 의도적으로 자신의 생명을 끊거나 자신의 신체를 의도적으로 해치는 행위를 묘사하거나 유도하는 발화
  S7(잘못된 정보): 개인이나 집단에게 잘못된 정보를 전파할 수 있는 발화

⚠ [세이프가드 즉시 폐기 규칙 — 모든 조건보다 최우선 적용]
세이프가드 태깅 결과에 다음 카테고리가 하나라도 포함된 경우:
  S3(성적 콘텐츠), S5(아동 성착취·그루밍), S6(자살·자해)
→ 동화의 전체 맥락, 교육적 의도와 무관하게 반드시 UNSAFE로 판정하세요.

[세이프가드 태깅 문장 주의사항 — S3/S5/S6 외 카테고리]
세이프가드 태깅 결과에 S1/S2/S4/S7 카테고리가 포함된 경우:
→ 전체 맥락이 교육적이더라도 태깅된 문장 자체를 주의깊게 검토하세요.
→ 태깅된 표현이 서사상 반드시 필요하지 않다면 UNSAFE로 판정하세요.

[SAFE 기준]
- 교육적 목적의 갈등 묘사 (밀었어요, 찢어졌어요, 울었어요 등)
- 반성·사과·화해로 이어지는 부정적 사건
- 동물 신체 손상 (날개 찢어짐 등) — 생명 존중 교훈 목적
- 역지사지 상상 표현 — 공감 유도 목적
- 권선징악 구조의 갈등 장면

[UNSAFE 기준]
- 욕설, 혐오 표현 (맥락 무관 즉시 UNSAFE)
- 직접적 폭력 위협
- 성적 암시·노출
- 그루밍·비밀 강요·신체 침해
- 반성·교훈 없이 폭력·가해만 묘사

[출력 형식] 반드시 아래 형식으로만 답하세요:
판정: SAFE 또는 UNSAFE
이유: <한 문장>"""


def _solar_user_msg(story_text: str, flagged_info: str = "") -> str:
    """Solar 사용자 메시지를 구성한다. pipeline.py의 EVAL_USER_TEMPLATE 형식과 동일."""
    msg = f"[동화 전문]\n{story_text}"
    if flagged_info:
        msg += f"\n\n[세이프가드 태깅 결과]\n{flagged_info}"
    else:
        msg += "\n\n[세이프가드 태깅 결과]\n없음"
    return msg


# ── Solar API ─────────────────────────────────────────────────────────────────

def call_solar(api_key: str, story_text: str, flagged_info: str = "",
               standalone: bool = False, max_retries: int = 5) -> str:
    """Solar API를 호출한다. standalone=True면 단순 적합성 판단, 아니면 세이프가드 규칙 포함."""
    if standalone:
        system   = SOLAR_SYSTEM_STANDALONE
        user_msg = f"[동화 전문]\n{story_text}"
    else:
        system   = SOLAR_SYSTEM
        user_msg = _solar_user_msg(story_text, flagged_info)
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                SOLAR_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": SOLAR_MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user_msg},
                    ],
                    "max_tokens": 100,
                    "temperature": 0.1,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError:
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"\n    ⏳ Rate limit — {wait}초 대기 ({attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Solar API {max_retries}회 재시도 실패")


def parse_solar_result(raw: str) -> str:
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("판정:"):
            val = line.replace("판정:", "").strip().upper()
            if "UNSAFE" in val:
                return "UNSAFE"
            if "SAFE" in val:
                return "SAFE"
    return "UNKNOWN"


def parse_solar_reason(raw: str) -> str:
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("이유:"):
            return line.replace("이유:", "").strip()
    return ""


def _get_solar_pred(val, default: str = "UNKNOWN") -> str:
    """solar_preds 값에서 예측 문자열을 꺼낸다 (구 형식 문자열과 신 형식 dict 모두 처리)."""
    if isinstance(val, dict):
        return val.get("pred", default)
    return val or default


def _get_solar_reason(val) -> str:
    """solar_preds 값에서 판정 이유를 꺼낸다."""
    if isinstance(val, dict):
        return val.get("reason", "")
    return ""


# ── Kanana safeguard ──────────────────────────────────────────────────────────

def load_kanana(use_adapter: bool):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    label  = "S5 어댑터 적용" if use_adapter else "베이스 (어댑터 없음)"
    print(f"  디바이스: {device} | 설정: {label}")

    tokenizer = AutoTokenizer.from_pretrained(KANANA_SAFEGUARD_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        KANANA_SAFEGUARD_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    if use_adapter:
        if os.path.exists(S5_ADAPTER_PATH):
            from peft import PeftModel
            print(f"  S5 어댑터 로딩: {S5_ADAPTER_PATH}")
            model = PeftModel.from_pretrained(model, S5_ADAPTER_PATH, local_files_only=True).eval()
            print("  어댑터 로딩 완료")
        else:
            print(f"  ⚠ 어댑터 없음 ({S5_ADAPTER_PATH}) — 베이스 모델로 대체")

    return model, tokenizer, device


def classify_sentence(model, tokenizer, device: str, sentence: str) -> dict:
    """
    단일 문장을 분류한다.

    Returns:
        {"pred": "SAFE"|"UNSAFE", "category": "S5"|None}
        pipeline.py의 safeguard.evaluate_story()가 반환하는 flagged 항목 형식과 동일.
    """
    import torch
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": sentence}],
        tokenize=True, return_tensors="pt",
        add_generation_prompt=True,
    ).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10,          # "<UNSAFE-S5>" 전체를 받기 위해 10토큰
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][input_ids.shape[-1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True).strip()

    if result.startswith("<UNSAFE"):
        m = re.search(r"UNSAFE-(S\d+)", result)
        category = m.group(1) if m else "S?"
        return {"pred": "UNSAFE", "category": category}
    return {"pred": "SAFE", "category": None}


def classify_by_pages(model, tokenizer, device: str, pages: list) -> dict:
    """페이지(단락) 단위로 하나씩 평가. 하나라도 UNSAFE면 즉시 반환."""
    for page in pages:
        text = page.get("srcText", "").strip()
        if not text:
            continue
        result = classify_sentence(model, tokenizer, device, text)
        if result["pred"] == "UNSAFE":
            return {**result, "sentence": text}
    return {"pred": "SAFE", "category": None, "sentence": ""}


def _compat_load(raw_val) -> dict:
    """구 체크포인트 형식(문자열)을 새 형식(dict)으로 변환한다."""
    if isinstance(raw_val, str):
        return {"pred": raw_val, "category": None}
    return raw_val


def run_kanana_pass(testset: list, use_adapter: bool, existing: dict,
                    page_eval: bool = False) -> dict:
    """
    카나나 세이프가드를 전체 테스트셋에 적용한다.

    page_eval=True  (기본 카나나 / 우리 모델): inserted_sentence 없으면 페이지 단위 평가
    page_eval=False: inserted_sentence 없으면 SAFE 반환

    Returns:
        {id: {"pred": "SAFE"|"UNSAFE", "category": "S5"|None}}
    """
    import torch

    # 구 형식 체크포인트 호환
    existing = {k: _compat_load(v) for k, v in existing.items()}

    items_todo = [x for x in testset if x["id"] not in existing]
    if not items_todo:
        label = "S5" if use_adapter else "기본"
        print(f"  ✅ {label} 카나나: 체크포인트에서 전체 복원 ({len(existing)}개)")
        return existing

    model, tokenizer, device = load_kanana(use_adapter=use_adapter)
    results = dict(existing)

    for i, item in enumerate(items_todo):
        sentence = item.get("inserted_sentence") or ""
        if sentence:
            info = classify_sentence(model, tokenizer, device, sentence)
            mode = "sentence"
        elif page_eval:
            pages = item.get("pages") or []
            info = classify_by_pages(model, tokenizer, device, pages)
            mode = f"pages({len(pages)})"
        else:
            info = {"pred": "SAFE", "category": None}
            mode = "skip"
        results[item["id"]] = info
        cat_str = f"/{info['category']}" if info["category"] else ""
        print(f"  [{i+1}/{len(items_todo)}] {item['id']} ({item['type']}) "
              f"[{mode}] → {info['pred']}{cat_str}", flush=True)

    del model
    torch.cuda.empty_cache()
    return results


def build_flagged_info(sentence: str, category: str) -> str:
    """
    세이프가드 태깅 결과를 Solar 프롬프트용 문자열로 변환한다.
    pipeline.py → evaluator.py의 flagged_info 포맷과 동일.
    """
    desc = CATEGORY_DESC.get(category, "기타")
    return f"  - {category}({desc}): {sentence}"


# ── test_data 로더 ────────────────────────────────────────────────────────────

def load_testset_from_dir(data_dir: str = TEST_DATA_DIR) -> list:
    """test_data/ 폴더의 JSON 파일을 testset 형식으로 변환한다."""
    items = []
    for filepath in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            name = os.path.splitext(os.path.basename(filepath))[0]
            gt = data.get("ground_truth", "SAFE").upper()
            paragraphs = sorted(data.get("paragraphInfo", []), key=lambda p: p.get("srcPage", 0))
            story_text = "\n".join(p.get("srcText", "") for p in paragraphs)
            items.append({
                "id":                name,
                "type":              gt,
                "ground_truth":      gt,
                "story_title":       data.get("title", ""),
                "story_text":        story_text,
                "pages":             paragraphs,
                "inserted_sentence": "",
                "source_file":       filepath,
            })
        except Exception as e:
            print(f"[TestData] {filepath} 로드 실패: {e}")
    return items


# ── 지표 계산 ──────────────────────────────────────────────────────────────────

def compute_metrics(results: list) -> dict:
    tp = fp = tn = fn = 0
    for r in results:
        gt, pred = r["ground_truth"], r["prediction"]
        if   gt == "UNSAFE" and pred == "UNSAFE": tp += 1
        elif gt == "SAFE"   and pred == "UNSAFE": fp += 1
        elif gt == "SAFE"   and pred == "SAFE":   tn += 1
        elif gt == "UNSAFE" and pred == "SAFE":   fn += 1

    total_safe   = tn + fp
    total_unsafe = tp + fn
    fpr       = fp / total_safe        if total_safe        > 0 else 0.0
    fnr       = fn / total_unsafe      if total_unsafe      > 0 else 0.0
    precision = tp / (tp + fp)         if (tp + fp)         > 0 else 0.0
    recall    = tp / (tp + fn)         if (tp + fn)         > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    acc       = (tp + tn) / (tp+fp+tn+fn) if (tp+fp+tn+fn) > 0 else 0.0

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Accuracy":  round(acc, 4),
        "Precision": round(precision, 4),
        "Recall":    round(recall, 4),
        "F1":        round(f1, 4),
        "FPR":       round(fpr, 4),
        "FNR":       round(fnr, 4),
    }


def compute_type_breakdown(results: list) -> dict:
    by_type: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        t = r["type"]
        by_type[t]["correct"] += int(r["ground_truth"] == r["prediction"])
        by_type[t]["total"]   += 1
    return {
        t: {
            "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0,
            "correct":  v["correct"],
            "total":    v["total"],
        }
        for t, v in sorted(by_type.items())
    }


# ── 출력 ──────────────────────────────────────────────────────────────────────

def print_results(systems: dict):
    names = list(systems.keys())
    w = 22

    print("\n" + "=" * (12 + w * len(names)))
    print("  📊 모델별 성능 비교")
    print("=" * (12 + w * len(names)))
    print(f"{'지표':<12}" + "".join(f"{n:>{w}}" for n in names))
    print("-" * (12 + w * len(names)))
    for key in ["Accuracy", "F1", "Precision", "Recall", "FPR", "FNR"]:
        print(f"{key:<12}" + "".join(
            f"{str(systems[n]['metrics'].get(key, '-')):>{w}}" for n in names))
    print("=" * (12 + w * len(names)))

    print("\n  혼동 행렬")
    print(f"  {'':12}" + "".join(f"{n:>{w}}" for n in names))
    for key in ["TP", "FP", "TN", "FN"]:
        print(f"  {key:<12}" + "".join(
            f"{str(systems[n]['metrics'].get(key, '-')):>{w}}" for n in names))

    print("\n  유형별 정확도  (A·C = UNSAFE, B·D = SAFE)")
    print(f"  {'유형':<8}" + "".join(f"{n:>{w}}" for n in names))
    for t in sorted({t for n in names for t in systems[n]["type_breakdown"]}):
        row = f"  {t:<8}"
        for n in names:
            bd  = systems[n]["type_breakdown"].get(t, {})
            acc = bd.get("accuracy", "-")
            c   = bd.get("correct", "-")
            tot = bd.get("total", "-")
            row += f"{f'{acc} ({c}/{tot})':>{w}}"
        print(row)
    print("=" * (12 + w * len(names)))

    base_m = systems["기본 카나나"]["metrics"]
    our_m  = systems["우리 모델"]["metrics"]
    print(f"\n  기본 카나나 → 우리 모델")
    for key, better in [("F1", "≥"), ("FPR", "≤"), ("FNR", "≤")]:
        bv, ov = base_m[key], our_m[key]
        improved = (ov >= bv) if better == "≥" else (ov <= bv)
        diff = round(ov - bv, 4)
        sign = f"+{diff}" if diff >= 0 else str(diff)
        print(f"    {key}: {bv} → {ov}  ({sign})  {'↑ 개선' if improved else '↓ 악화'}")
    print("=" * (12 + w * len(names)))


# ── 오답 분석 출력 ────────────────────────────────────────────────────────────

def print_wrong_predictions(our_results: list):
    """우리 모델이 틀린 항목과 최종 판정 경로·이유를 출력한다."""
    wrong = [
        r for r in our_results
        if r["ground_truth"] != r["prediction"]
        and r["prediction"] not in ("SKIP", "UNKNOWN")
    ]
    w = len(wrong)
    total = len([r for r in our_results if r["prediction"] not in ("SKIP", "UNKNOWN")])
    print(f"\n{'=' * 60}")
    print(f"  우리 모델 오답 분석 ({w}개 / 전체 {total}개)")
    print(f"{'=' * 60}")
    if not wrong:
        print("  ✅ 오답 없음")
        print("=" * 60)
        return
    for r in wrong:
        gt   = r["ground_truth"]
        pred = r["prediction"]
        err  = "FN (UNSAFE→SAFE)" if gt == "UNSAFE" else "FP (SAFE→UNSAFE)"
        print(f"\n  [{r['id']}]  GT={gt} / 예측={pred}  ← {err}")
        print(f"  제목    : {r.get('story_title', '-')}")
        print(f"  경로    : {r.get('pathway', '-')}")
        reason = r.get("solar_reason", "")
        print(f"  판정 이유  : {reason if reason else '(이유 없음)'}")
    print(f"\n{'=' * 60}")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="세 모델 시스템 성능 비교")
    parser.add_argument("--dry-run",     action="store_true", help="테스트셋 구조만 확인")
    parser.add_argument("--skip-kanana", action="store_true", help="Kanana 로딩 생략")
    args = parser.parse_args()

    if not os.path.isdir(TEST_DATA_DIR):
        print(f"❌ 테스트 데이터 폴더 없음: {TEST_DATA_DIR}")
        sys.exit(1)

    testset = load_testset_from_dir(TEST_DATA_DIR)

    type_counts = Counter(item["type"] for item in testset)
    gt_counts   = Counter(item["ground_truth"] for item in testset)
    print(f"✅ 테스트셋 로드: {len(testset)}개 항목")
    print(f"   유형별: {dict(sorted(type_counts.items()))}")
    print(f"   GT: SAFE={gt_counts['SAFE']}, UNSAFE={gt_counts['UNSAFE']}")

    if args.dry_run:
        print("\n[dry-run] 구조 확인 완료.")
        return

    api_key = os.getenv("UPSTAGE_API_KEY")
    solar_skip = not api_key
    if solar_skip:
        print("⚠ UPSTAGE_API_KEY 없음 — Solar 평가를 건너뜁니다.")

    # ── 체크포인트 복원 ───────────────────────────────────────────────────────
    ckpt: dict = {}
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            ckpt = json.load(f)
        print(f"✅ 체크포인트 복원: "
              f"base={len(ckpt.get('base_kanana', {}))}, "
              f"s5={len(ckpt.get('s5_kanana', {}))}, "
              f"solar={len(ckpt.get('solar', {}))}, "
              f"solar_flags={len(ckpt.get('solar_with_flags', {}))}")

    base_preds:        dict = ckpt.get("base_kanana", {})
    s5_preds:          dict = ckpt.get("s5_kanana", {})
    solar_preds:       dict = ckpt.get("solar", {})
    solar_flags_preds: dict = ckpt.get("solar_with_flags", {})

    def save_checkpoint():
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "base_kanana":      base_preds,
                "s5_kanana":        s5_preds,
                "solar":            solar_preds,
                "solar_with_flags": solar_flags_preds,
            }, f, ensure_ascii=False)

    # ── 1. 기본 카나나 (S5 어댑터 없음) ──────────────────────────────────────
    if not args.skip_kanana:
        print("\n⏳ [1/4] 기본 카나나 세이프가드 (S5 어댑터 없음)")
        base_preds = run_kanana_pass(testset, use_adapter=False, existing=base_preds, page_eval=True)
        save_checkpoint()
        print(f"✅ 기본 카나나 완료 ({len(base_preds)}개)")

    # ── 2. S5 카나나 (우리 모델 1단계) ───────────────────────────────────────
    if not args.skip_kanana:
        print("\n⏳ [2/4] S5 카나나 (S5 어댑터 적용)")
        s5_preds = run_kanana_pass(testset, use_adapter=True, existing=s5_preds, page_eval=True)
        save_checkpoint()
        print(f"✅ S5 카나나 완료 ({len(s5_preds)}개)")

    # ── 3. Solar 단독 (세이프가드 정보 없음) ─────────────────────────────────
    if not solar_skip:
        items_todo = [x for x in testset if x["id"] not in solar_preds]
        if items_todo:
            print(f"\n⏳ [3/4] Solar 단독 평가 ({len(items_todo)}개)")
            for i, item in enumerate(items_todo):
                raw    = call_solar(api_key, item["story_text"], standalone=True)
                pred   = parse_solar_result(raw)
                reason = parse_solar_reason(raw)
                solar_preds[item["id"]] = {"pred": pred, "reason": reason}
                print(f"  [{i+1}/{len(items_todo)}] {item['id']} ({item['type']}) → {pred}",
                      flush=True)
                if (i + 1) % 10 == 0:
                    save_checkpoint()
                    print("    💾 체크포인트 저장")
                time.sleep(1.5)
            print(f"✅ Solar 단독 완료 ({len(solar_preds)}개)")
        else:
            print(f"\n✅ [3/4] Solar 단독: 체크포인트 복원 ({len(solar_preds)}개)")

    # ── 4. Solar + 세이프가드 태그 (우리 모델 2단계) ─────────────────────────
    # S5 카나나가 UNSAFE로 판정한 항목은 카테고리와 무관하게 Solar에 태그 정보를 전달한다.
    # Solar 프롬프트의 즉시 폐기 규칙(S3/S5/S6)은 Solar가 직접 적용한다.
    if not solar_skip and not args.skip_kanana:
        items_need_solar_flags = [
            x for x in testset
            if x["id"] not in solar_flags_preds
            and _compat_load(s5_preds.get(x["id"], {"pred": "SKIP"}))["pred"] == "UNSAFE"
        ]
        if items_need_solar_flags:
            print(f"\n⏳ [4/4] Solar + 세이프가드 태그 전달 ({len(items_need_solar_flags)}개)")
            for i, item in enumerate(items_need_solar_flags):
                iid  = item["id"]
                info = _compat_load(s5_preds[iid])
                flagged_sentence = (
                    info.get("sentence")
                    or item.get("inserted_sentence", "")
                )
                flagged_info = build_flagged_info(
                    flagged_sentence,
                    info["category"] or "S?",
                )
                raw    = call_solar(api_key, item["story_text"], flagged_info=flagged_info)
                pred   = parse_solar_result(raw)
                reason = parse_solar_reason(raw)
                solar_flags_preds[iid] = {"pred": pred, "reason": reason}
                print(f"  [{i+1}/{len(items_need_solar_flags)}] {iid} "
                      f"({info['category']}) → {pred}", flush=True)
                if (i + 1) % 10 == 0:
                    save_checkpoint()
                    print("    💾 체크포인트 저장")
                time.sleep(1.5)
            print(f"✅ Solar+플래그 완료 ({len(solar_flags_preds)}개)")
        else:
            print(f"\n✅ [4/4] Solar+플래그: 체크포인트 복원 또는 해당 항목 없음")

    # ── 결과 조립 ─────────────────────────────────────────────────────────────
    base_results  = []
    solar_results = []
    our_results   = []

    for item in testset:
        iid = item["id"]
        gt = item["ground_truth"]
        common = {
            "id":                iid,
            "type":              item["type"],
            "ground_truth":      gt,
            "story_title":       item.get("story_title", ""),
            "inserted_sentence": item.get("inserted_sentence", ""),
        }

        base_info  = _compat_load(base_preds.get(iid, {"pred": "SKIP"}))
        s5_info    = _compat_load(s5_preds.get(iid,   {"pred": "SKIP"}))
        base_pred  = base_info["pred"]
        s5_pred    = s5_info["pred"]
        s5_cat     = s5_info.get("category")

        _solar_val = solar_preds.get(iid)
        solar_pred = _get_solar_pred(_solar_val)

        # 우리 모델 판정 — pipeline.py + evaluator.py와 동일한 흐름
        # S5 카나나 UNSAFE → Solar에 태그 전달 (Solar가 S3/S5/S6 규칙 직접 적용)
        # S5 카나나 SAFE   → Solar 단독 판정 (태그 없음)
        if s5_pred == "UNSAFE":
            # 태그 정보를 받은 Solar가 판정 (S3/S5/S6이면 Solar가 무조건 UNSAFE 반환)
            _flags_val  = solar_flags_preds.get(iid)
            our_pred    = _get_solar_pred(_flags_val) if _flags_val is not None else solar_pred
            our_reason  = _get_solar_reason(_flags_val) if _flags_val is not None else _get_solar_reason(_solar_val)
            our_pathway = f"Solar+플래그({s5_cat})"
        elif s5_pred == "SAFE":
            # 세이프가드 통과 → Solar 단독 판정
            our_pred    = solar_pred
            our_reason  = _get_solar_reason(_solar_val)
            our_pathway = "Solar단독"
        else:
            our_pred    = solar_pred  # SKIP fallback
            our_reason  = _get_solar_reason(_solar_val)
            our_pathway = "fallback"

        base_results.append( {**common, "prediction": base_pred,
                               "category": base_info.get("category")})
        solar_results.append({**common, "prediction": solar_pred})
        our_results.append(  {**common, "prediction": our_pred,
                               "s5_pred": s5_pred, "s5_category": s5_cat,
                               "solar_pred": solar_pred, "pathway": our_pathway,
                               "solar_reason": our_reason})

    def valid(results):
        return [r for r in results if r["prediction"] not in ("SKIP", "UNKNOWN")]

    systems = {
        "기본 카나나": {
            "metrics":        compute_metrics(valid(base_results)),
            "type_breakdown": compute_type_breakdown(valid(base_results)),
            "results":        base_results,
        },
        "Solar 단독": {
            "metrics":        compute_metrics(valid(solar_results)),
            "type_breakdown": compute_type_breakdown(valid(solar_results)),
            "results":        solar_results,
        },
        "우리 모델": {
            "metrics":        compute_metrics(valid(our_results)),
            "type_breakdown": compute_type_breakdown(valid(our_results)),
            "results":        our_results,
        },
    }

    print_results(systems)
    print_wrong_predictions(our_results)

    output = {
        "testset_summary": {
            "total":   len(testset),
            "by_type": dict(sorted(type_counts.items())),
            "by_gt":   dict(gt_counts),
        },
        "summary": {
            name: {"metrics": v["metrics"], "type_breakdown": v["type_breakdown"]}
            for name, v in systems.items()
        },
        "details": {name: v["results"] for name, v in systems.items()},
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    print(f"\n  결과 저장: {RESULT_PATH}")


if __name__ == "__main__":
    main()
