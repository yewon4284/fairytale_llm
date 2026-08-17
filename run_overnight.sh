#!/bin/bash
# run_overnight.sh
# 자는 동안 무인으로 순차 실행:
#   0) pod 셋업 스모크테스트 (GPU/데이터/API키 확인)
#   1) Solar Pro3/Pro4/HCX-007 미니 스모크테스트 (본 실행 전에 API가 실제로 도는지 확인)
#   2) base(기존 어댑터) Solar Pro3 441편 -- held-out 비교의 "재보정 전" 기준선 겸
#      3자 비교용 결과로 재사용 (2026-08-18: eval_results_base.json이 새 pod엔 아예
#      없어서 예전 순서로는 held-out 비교가 매번 조용히 스킵되던 문제를 고침)
#   3) held-out 분할 기준 재보정 어댑터 재학습 + held-out 20%로 검증
#      (13단계는 --solar-model solar-pro3로 명시 고정 -- 2단계와 동일 모델임을
#      보장해서 "어댑터 차이"만 순수하게 비교되게 함)
#   4) Solar Pro4 / HCX-007 441편 (Pro3는 2단계에서 이미 실행함)
#   5) 최종 3자 비교 리포트
#
# 설계 원칙:
#   - set -e 안 씀. 한 단계가 실패해도 스크립트 전체가 죽지 않고 다음 단계로 넘어감
#     (자는 동안 한 곳에서 멈춰서 나머지가 하나도 안 도는 상황 방지)
#   - 각 단계 로그는 overnight_logs/에 개별 저장, 성공/실패 요약은 overnight_status.md에 누적
#   - eval_test_dataset.py는 이미 파일 단위로 이어서 실행되는 기능이 있어서
#     (load_existing으로 완료분 스킵) SSH 끊김/pod 재시작에도 중간부터 재개 가능.
#     스크립트가 죽었으면 그냥 다시 `bash run_overnight.sh`만 실행하면 됨 — 이미 끝난 단계는
#     각 python 스크립트 자체의 존재 체크(예: output json이 이미 441개 다 있으면 사실상
#     바로 끝남) 덕분에 크게 낭비 없이 이어감.
#   - HCX는 API 응답 형식이 미검증이라, 스모크테스트에서 실패하면 본 실행에서 자동으로
#     건너뛰고 나머지(Pro3 vs Pro4 비교)는 정상 진행됨.
#
# 예상 소요시간: 스토리당 세이프가드+API 호출 합쳐 1~2분 잡으면, 441편짜리 실행 한 번에
#   약 7~15시간. 이 스크립트는 총 4번의 441편 전체 실행(held-out 재보정 검증 1 +
#   Pro3/Pro4/HCX 비교 3)을 순서대로 하므로 하룻밤에 다 안 끝날 수 있음 — 괜찮음,
#   아침에 morning-run_overnight.sh를 다시 실행하면 이어서 진행됨.
#
# 사용법:
#   nohup bash run_overnight.sh > overnight_main.log 2>&1 &
#   disown
#   (다음날 아침) cat overnight_status.md

set -uo pipefail

LOGDIR="overnight_logs"
mkdir -p "$LOGDIR"
STATUS_FILE="overnight_status.md"
echo "# 밤새 실행 로그 시작: $(date)" >> "$STATUS_FILE"
echo "" >> "$STATUS_FILE"

log_status() {
    echo "$1" | tee -a "$STATUS_FILE"
}

run_step() {
    local name="$1"; shift
    echo "================================================================"
    echo "[$name] 시작: $(date)"
    if "$@" > "$LOGDIR/${name}.log" 2>&1; then
        log_status "- [$(date +%H:%M)] ✅ $name 성공"
        return 0
    else
        log_status "- [$(date +%H:%M)] ❌ $name 실패 — 로그: $LOGDIR/${name}.log"
        tail -20 "$LOGDIR/${name}.log" | sed 's/^/    /' >> "$STATUS_FILE"
        return 1
    fi
}

echo "================================================================"
echo "  0/4. 사전 점검 (여기서 막히면 뒤에 아무것도 못 함 — 바로 알림)"
echo "================================================================"

DATA_COUNT=$(ls all_data/*.json 2>/dev/null | wc -l)
if [ "$DATA_COUNT" -ne 441 ]; then
    log_status "❌ 치명적: all_data가 441개가 아니라 ${DATA_COUNT}개임. scp로 올렸는지 확인 필요."
    log_status "   (로컬 PC PowerShell에서: & \"C:\\Windows\\System32\\OpenSSH\\scp.exe\" -P 30154 -i \$env:USERPROFILE\\.ssh\\id_ed25519 -r \"C:\\Users\\jyoun\\Desktop\\linux\\fairytale_llm\\all_data\" root@38.147.83.14:~/fairytale_llm/)"
    log_status "   all_data 없이는 이후 단계가 전부 실패하므로 여기서 스크립트를 종료합니다."
    exit 1
fi
log_status "✅ all_data 441개 확인됨"

if [ ! -f "kanana_solar_eval_441.jsonl" ]; then
    log_status "❌ 치명적: kanana_solar_eval_441.jsonl 없음 (git pull 확인 필요). 종료합니다."
    exit 1
fi
log_status "✅ kanana_solar_eval_441.jsonl 확인됨"

if ! grep -q "UPSTAGE_API_KEY" .env 2>/dev/null; then
    log_status "❌ 치명적: .env에 UPSTAGE_API_KEY 없음. 종료합니다."
    exit 1
fi
log_status "✅ UPSTAGE_API_KEY 확인됨"

HCX_AVAILABLE=1
if ! grep -q "HCX_API_KEY" .env 2>/dev/null; then
    log_status "⚠ .env에 HCX_API_KEY 없음 — HCX-007 비교는 건너뛰고 Solar Pro3 vs Pro4만 진행합니다."
    HCX_AVAILABLE=0
fi

run_step "01_smoke_gpu" python3 -c "import torch; assert torch.cuda.is_available(), 'GPU 없음'; print(torch.cuda.get_device_name(0))"
if [ $? -ne 0 ]; then
    log_status "❌ 치명적: GPU 확인 실패. bash setup_pod.sh부터 다시 실행 필요. 종료합니다."
    exit 1
fi

echo ""
echo "================================================================"
echo "  1/4. 평가 백엔드 스모크테스트 (본 실행 전 2편만 먼저 확인)"
echo "================================================================"

run_step "02_smoke_solar_pro3" python eval_test_dataset.py --limit 2 --evaluator solar --solar-model solar-pro3 --output "$LOGDIR/smoke_pro3.json"
run_step "03_smoke_solar_pro4" python eval_test_dataset.py --limit 2 --evaluator solar --solar-model solar-pro4 --output "$LOGDIR/smoke_pro4.json"

if [ "$HCX_AVAILABLE" = "1" ]; then
    if run_step "04_smoke_hcx" python eval_test_dataset.py --limit 2 --evaluator hcx --output "$LOGDIR/smoke_hcx.json"; then
        log_status "✅ HCX 스모크테스트 통과 — 본 실행에 포함합니다"
    else
        log_status "⚠ HCX 스모크테스트 실패 — src/evaluator_hcx.py의 응답 파싱을 실제 API 형식에 맞게 고쳐야 함. 본 실행에서는 건너뜁니다."
        HCX_AVAILABLE=0
    fi
fi

echo ""
echo "================================================================"
echo "  2/4. base(기존 어댑터) Solar Pro3 441편 실행"
echo "  -- held-out 비교의 '재보정 전' 기준선으로도 재사용하고, 3자 비교에도"
echo "     그대로 쓴다(eval_results_base.json을 새 pod마다 따로 준비할 필요가"
echo "     없어짐 -- 2026-08-18에 새 pod엔 그 파일이 아예 없다는 걸 발견해서 순서를"
echo "     이렇게 재배치함, 원래는 held-out 비교(옛 14단계)가 뒤에 있어서 매번"
echo "     조용히 건너뛰기만 하고 있었음)."
echo "================================================================"

run_step "20_eval_solar_pro3" python eval_test_dataset.py --include-fewshot \
    --evaluator solar --solar-model solar-pro3 --output eval_results_solar_pro3.json

echo ""
echo "================================================================"
echo "  3/4. held-out 분할 기준 재보정 어댑터 재학습 + 검증"
echo "================================================================"

run_step "10_build_calib_dataset" python finetune/build_calibration_dataset.py \
    --eval-jsonl kanana_solar_eval_441.jsonl --corpus-dir all_data \
    --output finetune/calibration_dataset_clean.json \
    --exclude-files finetune/holdout_files.json \
    --merge-with finetune/s5_dataset.json

run_step "11_mine_domain_safe" python finetune/mine_domain_safe_examples.py \
    --eval-jsonl kanana_solar_eval_441.jsonl --corpus-dir all_data \
    --output finetune/calibration_dataset_clean.json \
    --merge-with finetune/calibration_dataset_clean.json \
    --exclude-files finetune/holdout_files.json --per-group-limit 40

run_step "12_train_adapter" python finetune/train_s5.py \
    --dataset finetune/calibration_dataset_clean.json \
    --output finetune/kanana-calibration-adapter-clean

# --solar-model을 명시적으로 solar-pro3로 고정 -- 안 고정하면 evaluator.py 기본값
# (모듈 상수 SOLAR_MODEL="solar-pro")을 쓰게 되는데, 이게 20단계의 "solar-pro3"와
# 실제로 동일 모델인지 100% 확인된 바가 없음. 같은 문자열을 강제해서 재보정 전/후
# 비교가 "어댑터 차이"만 순수하게 반영하도록 함 (안 그러면 모델 문자열 차이까지
# 섞여서 뭐 때문에 달라졌는지 알 수 없게 됨 -- 예전 퓨샷 고정 실수와 같은 종류의 함정).
run_step "13_eval_calibrated_clean" python eval_test_dataset.py --include-fewshot \
    --adapter-path finetune/kanana-calibration-adapter-clean/final_adapter \
    --evaluator solar --solar-model solar-pro3 \
    --output eval_results_calibrated_clean.json

run_step "14_compare_holdout" python compare_calibrated_run.py \
    --ground-truth kanana_solar_eval_441.jsonl \
    --before eval_results_solar_pro3.json --after eval_results_calibrated_clean.json \
    --restrict-files finetune/holdout_files.json

echo ""
echo "================================================================"
echo "  4/5. Solar Pro4 / HCX-007 (20단계 solar-pro3는 위에서 이미 실행함)"
echo "================================================================"

run_step "21_eval_solar_pro4" python eval_test_dataset.py --include-fewshot \
    --evaluator solar --solar-model solar-pro4 --output eval_results_solar_pro4.json

if [ "$HCX_AVAILABLE" = "1" ]; then
    run_step "22_eval_hcx007" python eval_test_dataset.py --include-fewshot \
        --evaluator hcx --output eval_results_hcx007.json
fi

echo ""
echo "================================================================"
echo "  5/5. 최종 비교 리포트"
echo "================================================================"

COMPARE_ARGS="--ground-truth kanana_solar_eval_441.jsonl --result solar_pro3=eval_results_solar_pro3.json --result solar_pro4=eval_results_solar_pro4.json"
if [ "$HCX_AVAILABLE" = "1" ] && [ -f "eval_results_hcx007.json" ]; then
    COMPARE_ARGS="$COMPARE_ARGS --result hcx007=eval_results_hcx007.json"
fi
run_step "23_compare_backends" python compare_evaluator_backends.py $COMPARE_ARGS

echo "" >> "$STATUS_FILE"
echo "## 전체 완료: $(date)" >> "$STATUS_FILE"
echo "================================================================"
echo "밤새 실행 끝. 아래 파일들 확인하세요:"
echo "  cat overnight_status.md          # 단계별 성공/실패 요약"
echo "  cat overnight_logs/23_compare_backends.log   # 최종 3자 비교 결과"
