#!/bin/bash
# watchdog.sh — run_ab_test.py를 감시하면서, GPU 메모리 부족으로 CPU 폴백이 감지되면
# 자동으로 죽이고 재시작한다. run_ab_test.py는 이어서-재개 구조라 중간에 죽여도 안전하다.
#
# 사용법:
#   chmod +x watchdog.sh
#   nohup ./watchdog.sh > watchdog.log 2>&1 &
#   disown
#
# run_ab_test.py에 넘길 인자는 아래 CMD 줄을 직접 수정하세요.

set -u
LOG="run_watchdog_$(date +%s).log"
CMD="python run_ab_test.py --extra-condition cat_의사소통 --extra-fewshot-dir data_sorted_cat_의사소통 --extra-condition cat_예술경험 --extra-fewshot-dir data_sorted_cat_예술경험"
CHECK_INTERVAL=60   # 몇 초마다 로그를 확인할지

echo "[watchdog] 시작. 로그 파일: $LOG"

while true; do
    echo "[watchdog] $(date '+%H:%M:%S') 프로세스 시작: $CMD"
    $CMD >> "$LOG" 2>&1 &
    PID=$!
    echo "[watchdog] PID $PID"

    RESTART_NEEDED=0
    while kill -0 "$PID" 2>/dev/null; do
        sleep "$CHECK_INTERVAL"

        if ! kill -0 "$PID" 2>/dev/null; then
            break
        fi

        if tail -n 100 "$LOG" | grep -q "CPU 재로딩"; then
            echo "[watchdog] $(date '+%H:%M:%S') GPU 메모리 부족(CPU 폴백) 감지 -> 프로세스 강제 종료 후 재시작"
            kill -9 "$PID" 2>/dev/null
            sleep 10
            RESTART_NEEDED=1
            break
        fi
    done

    wait "$PID" 2>/dev/null
    EXIT_CODE=$?

    if grep -q "전체 완료" "$LOG"; then
        echo "[watchdog] $(date '+%H:%M:%S') '전체 완료' 확인됨 — 모든 생성 끝. watchdog 종료."
        break
    fi

    if [ "$RESTART_NEEDED" -eq 1 ]; then
        echo "[watchdog] CPU 폴백으로 인한 재시작. 잠시 대기 후 재시작합니다."
        sleep 5
        continue
    fi

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "[watchdog] $(date '+%H:%M:%S') 프로세스가 에러로 종료됨(exit=$EXIT_CODE). 5초 후 재시작합니다."
        sleep 5
        continue
    fi

    echo "[watchdog] 프로세스가 정상 종료됐지만 '전체 완료' 문구가 없습니다. 로그를 확인하세요: $LOG"
    break
done

echo "[watchdog] 종료."
