"""
compare_evaluator_backends.py
Solar Pro3 / Solar Pro4 / HCX-007 등 2차 평가 백엔드를 몇 개든 한 번에 비교한다.

전제: 각 백엔드 결과는 eval_test_dataset.py --evaluator {solar,hcx} --solar-model ...
로 "같은 evaluator.py 프롬프트·같은 세이프가드"로 돌린 eval_test_dataset.py 결과 json이어야
모델 차이만 순수하게 비교할 수 있다 (프롬프트가 다르면 무엇 때문에 차이났는지 알 수 없음).

사용법:
    python compare_evaluator_backends.py --ground-truth kanana_solar_eval_441.jsonl \
        --result solar_pro3=eval_results_solar_pro3.json \
        --result solar_pro4=eval_results_solar_pro4.json \
        --result hcx007=eval_results_hcx007.json
"""

import argparse
import json
import sys

from eval_compare import confusion_metrics, mcnemar_exact, norm_label


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return {r.get("file") or r.get("filename"): r for r in rows}


def load_eval_json(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return {r["filename"]: r for r in rows}


def parse_kv(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            print(f"--result는 name=path 형식이어야 합니다: {item}")
            sys.exit(1)
        name, path = item.split("=", 1)
        out[name] = path
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", default="kanana_solar_eval_441.jsonl")
    ap.add_argument("--result", action="append", required=True,
                     help="name=path.json 형식, 여러 번 지정 가능 (2개 이상 필요)")
    ap.add_argument("--restrict-files", default=None,
                     help="특정 파일 목록으로만 채점 (예: finetune/holdout_files.json)")
    args = ap.parse_args()

    backends = parse_kv(args.result)
    if len(backends) < 2:
        print("최소 2개 이상의 --result가 필요합니다.")
        sys.exit(1)

    gt = load_jsonl(args.ground_truth)
    loaded = {name: load_eval_json(path) for name, path in backends.items()}

    restrict = None
    if args.restrict_files:
        with open(args.restrict_files, encoding="utf-8") as f:
            restrict = set(json.load(f))

    # 모든 백엔드에서 공통으로 결과가 있는 파일만 (공정한 비교를 위해)
    common_files = set(gt.keys())
    for name, results in loaded.items():
        common_files &= set(results.keys())
    if restrict:
        common_files &= restrict

    filtered = [f for f in gt if gt[f].get("ground_truth") is not None]
    common_files &= set(filtered)

    print(f"ground_truth {len(gt)}편 / 백엔드별 결과 있음 / 공통 채점 가능 편수: {len(common_files)}\n")
    if not common_files:
        print("공통으로 채점 가능한 파일이 없습니다. --result 경로들을 확인하세요.")
        sys.exit(1)

    # ── 백엔드별 accuracy/precision/recall/F1 ────────────────────────────────
    print("=" * 70)
    print("  백엔드별 성능 (정답 대비)")
    print("=" * 70)
    correctness = {}  # name -> {file: bool}
    metrics = {}
    for name, results in loaded.items():
        rows = []
        correct = {}
        for f in common_files:
            pred = results[f].get("verdict")
            truth = gt[f].get("ground_truth")
            rows.append({"pred": pred, "gt": truth})
            correct[f] = (norm_label(pred) == norm_label(truth))
        m = confusion_metrics(rows, "pred", "gt")
        metrics[name] = m
        correctness[name] = correct
        print(f"\n  [{name}]")
        print(f"    정확도 {m['acc']:.4f} / 정밀도 {m['prec']:.4f} / "
              f"재현율 {m['rec']:.4f} / F1 {m['f1']:.4f}")
        print(f"    tp={m['tp']} fp={m['fp']} tn={m['tn']} fn={m['fn']}")

    # ── 순위 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  F1 기준 순위")
    print("=" * 70)
    ranked = sorted(metrics.items(), key=lambda kv: kv[1]["f1"], reverse=True)
    for i, (name, m) in enumerate(ranked, 1):
        print(f"  {i}위: {name} (F1={m['f1']:.4f}, acc={m['acc']:.4f})")

    # ── pairwise McNemar ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  쌍별 McNemar exact test (정확도 기준)")
    print("=" * 70)
    names = list(loaded.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            vec_a = [correctness[a][f] for f in common_files]
            vec_b = [correctness[b][f] for f in common_files]
            b_only, c_only, p = mcnemar_exact(vec_a, vec_b)
            sig = "유의미(p<0.05)" if p < 0.05 else "유의미하지 않음"
            better = a if b_only > c_only else (b if c_only > b_only else "동률")
            print(f"  {a} vs {b}: {a}만 맞음={b_only} / {b}만 맞음={c_only} / "
                  f"p={p:.4f} ({sig}, 우세: {better})")


if __name__ == "__main__":
    main()
