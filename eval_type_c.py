"""
eval_type_c.py

테스트셋 C 유형(S8 그루밍 패턴, 모두 UNSAFE)에서
기본 카나나 vs S8 학습 카나나의 탐지 성능을 비교한다.

C 유형은 기본 카나나가 미탐지하도록 설계된 항목들이므로,
S8 어댑터 학습 효과를 직접적으로 측정하는 데 적합하다.

주요 지표:
  탐지율(Recall) — 전체 UNSAFE 중 UNSAFE로 정확히 잡은 비율
  FNR            — 놓친 비율 (낮을수록 좋음)

실행:
  python eval_type_c.py
  python eval_type_c.py --dry-run
"""

import argparse
import json
import os
import sys

TESTSET_PATH = os.path.join(os.path.dirname(__file__), "testset.json")
RESULT_PATH  = os.path.join(os.path.dirname(__file__), "eval_type_c_results.json")

KANANA_MODEL_ID = "kakaocorp/kanana-safeguard-8b"
S8_ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "finetune/kanana-s8-adapter/final_adapter")


# ── 모델 로드 / 분류 ──────────────────────────────────────────────────────────

def load_kanana(use_adapter: bool):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    label  = "S8 어댑터 적용" if use_adapter else "베이스 (어댑터 없음)"
    print(f"  디바이스: {device} | 설정: {label}")

    tokenizer = AutoTokenizer.from_pretrained(KANANA_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        KANANA_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    if use_adapter:
        if os.path.exists(S8_ADAPTER_PATH):
            from peft import PeftModel
            print(f"  S8 어댑터 로딩: {S8_ADAPTER_PATH}")
            model = PeftModel.from_pretrained(model, S8_ADAPTER_PATH).eval()
            print("  어댑터 로딩 완료")
        else:
            print(f"  ⚠ 어댑터 없음 ({S8_ADAPTER_PATH}) — 베이스 모델로 대체")

    return model, tokenizer, device


def classify(model, tokenizer, device: str, sentence: str) -> str:
    import torch
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": sentence}],
        tokenize=True, return_tensors="pt",
    ).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=1, pad_token_id=tokenizer.eos_token_id,
        )
    token = tokenizer.decode(output_ids[0][input_ids.shape[-1]], skip_special_tokens=True)
    return "UNSAFE" if token.startswith("<UNSAFE") else "SAFE"


def run_eval(items: list, model, tokenizer, device: str, label: str) -> list:
    results = []
    for i, item in enumerate(items):
        sentence = item.get("inserted_sentence", "")
        pred = classify(model, tokenizer, device, sentence) if sentence else "SAFE"
        correct = (pred == item["ground_truth"])
        results.append({
            "id":               item["id"],
            "description":      item.get("description", ""),
            "inserted_sentence": sentence,
            "ground_truth":     item["ground_truth"],
            "prediction":       pred,
            "correct":          correct,
        })
        mark = "✅" if correct else "❌"
        print(f"  [{i+1:02d}/{len(items)}] {item['id']}  {mark}  pred={pred}  "
              f"| {sentence[:45]}{'...' if len(sentence) > 45 else ''}")
    return results


# ── 지표 계산 ─────────────────────────────────────────────────────────────────

def metrics(results: list) -> dict:
    """C 유형은 전부 UNSAFE이므로 TP/FN만 존재."""
    tp = sum(1 for r in results if r["prediction"] == "UNSAFE")
    fn = sum(1 for r in results if r["prediction"] == "SAFE")
    total = tp + fn
    recall = tp / total if total > 0 else 0.0
    fnr    = fn / total if total > 0 else 0.0
    return {
        "total":    total,
        "detected": tp,
        "missed":   fn,
        "Recall":   round(recall, 4),
        "FNR":      round(fnr, 4),
    }


# ── 출력 ──────────────────────────────────────────────────────────────────────

def print_summary(base_results: list, s8_results: list):
    bm = metrics(base_results)
    sm = metrics(s8_results)

    w = 20
    print("\n" + "=" * (14 + w * 2))
    print("  📊 C 유형 탐지 성능 비교  (전체 UNSAFE)")
    print("=" * (14 + w * 2))
    print(f"{'지표':<14}{'기본 카나나':>{w}}{'S8 학습 카나나':>{w}}")
    print("-" * (14 + w * 2))
    for key in ["total", "detected", "missed", "Recall", "FNR"]:
        bv = str(bm[key])
        sv = str(sm[key])
        print(f"{key:<14}{bv:>{w}}{sv:>{w}}")
    print("=" * (14 + w * 2))

    # 개선 요약
    recall_diff = round(sm["Recall"] - bm["Recall"], 4)
    fnr_diff    = round(sm["FNR"]    - bm["FNR"],    4)
    print(f"\n  기본 → S8 학습 카나나")
    print(f"    탐지율(Recall): {bm['Recall']} → {sm['Recall']}  "
          f"({'↑ +' if recall_diff >= 0 else '↓ '}{abs(recall_diff)})")
    print(f"    FNR:           {bm['FNR']} → {sm['FNR']}  "
          f"({'↓ -' if fnr_diff <= 0 else '↑ +'}{abs(fnr_diff)})")

    # 항목별 비교
    print("\n  항목별 비교  (B=기본 카나나  S=S8 카나나  ✅=UNSAFE 정탐  ❌=SAFE 미탐)")
    print(f"  {'ID':<10} {'B':^6} {'S':^6}  설명")
    print("  " + "-" * 68)
    base_map = {r["id"]: r for r in base_results}
    for r in s8_results:
        iid = r["id"]
        b_pred = base_map[iid]["prediction"]
        s_pred = r["prediction"]
        b_mark = "✅" if b_pred == "UNSAFE" else "❌"
        s_mark = "✅" if s_pred == "UNSAFE" else "❌"
        # S8이 잡았는데 기본이 못 잡은 경우 강조
        highlight = " ◀ S8 추가 탐지" if (b_pred == "SAFE" and s_pred == "UNSAFE") else ""
        highlight += " ◀ 둘 다 탐지" if (b_pred == "UNSAFE" and s_pred == "UNSAFE") else ""
        highlight += " ◀ 둘 다 미탐" if (b_pred == "SAFE"   and s_pred == "SAFE")   else ""
        print(f"  {iid:<10} {b_mark:^6} {s_mark:^6}  {r['description'][:45]}{highlight}")
    print("=" * (14 + w * 2))


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="C 유형 S8 탐지 성능 비교")
    parser.add_argument("--dry-run", action="store_true", help="테스트셋 구조만 확인")
    args = parser.parse_args()

    if not os.path.exists(TESTSET_PATH):
        print(f"❌ 테스트셋 없음: {TESTSET_PATH}")
        sys.exit(1)

    with open(TESTSET_PATH, encoding="utf-8") as f:
        testset = json.load(f)

    c_items = [x for x in testset if x["type"] == "C"]
    print(f"✅ C 유형 항목: {len(c_items)}개 (전체 {len(testset)}개 중)")
    print(f"   GT: UNSAFE={sum(1 for x in c_items if x['ground_truth']=='UNSAFE')}, "
          f"SAFE={sum(1 for x in c_items if x['ground_truth']=='SAFE')}")

    if args.dry_run:
        print("\n[dry-run] 구조 확인 완료.")
        for x in c_items:
            print(f"  {x['id']}: {x['description']}")
        return

    import torch

    # ── 1. 기본 카나나 ────────────────────────────────────────────────────────
    print("\n⏳ [1/2] 기본 카나나 세이프가드 (S8 어댑터 없음)")
    model, tokenizer, device = load_kanana(use_adapter=False)
    base_results = run_eval(c_items, model, tokenizer, device, "기본 카나나")
    del model
    torch.cuda.empty_cache()
    print(f"✅ 완료: 탐지 {metrics(base_results)['detected']}/{len(c_items)}")

    # ── 2. S8 학습 카나나 ─────────────────────────────────────────────────────
    print("\n⏳ [2/2] S8 학습 카나나 (S8 어댑터 적용)")
    model, tokenizer, device = load_kanana(use_adapter=True)
    s8_results = run_eval(c_items, model, tokenizer, device, "S8 카나나")
    del model
    torch.cuda.empty_cache()
    print(f"✅ 완료: 탐지 {metrics(s8_results)['detected']}/{len(c_items)}")

    # ── 결과 출력 & 저장 ──────────────────────────────────────────────────────
    print_summary(base_results, s8_results)

    output = {
        "type_c_count":  len(c_items),
        "base_kanana":   {"metrics": metrics(base_results),  "results": base_results},
        "s8_kanana":     {"metrics": metrics(s8_results),    "results": s8_results},
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {RESULT_PATH}")


if __name__ == "__main__":
    main()
