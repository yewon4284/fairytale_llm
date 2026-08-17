#!/bin/bash
# setup_pod.sh
# 새 RunPod 인스턴스에서 딱 한 번 실행하면 지금까지 겪었던 반복 문제들을 미리 처리한다:
#   - torch가 기본으로 CUDA 13 런타임을 깔아서 드라이버(보통 12.4)와 안 맞아 GPU를 못 잡는 문제
#   - HuggingFace 캐시가 작은 컨테이너 디스크(/)에 받아져서 용량 부족으로 다운로드 실패하는 문제
#   - trl 최신 버전이 DataCollatorForCompletionOnlyLM을 제거해서 train_s5.py가 깨지는 문제
#     (requirements.txt에 trl==0.19.1로 이미 고정해뒀으니 이 스크립트에서 별도 처리 불필요)
#
# 자동으로 못 하는 것 (수동으로 해야 함):
#   - .env에 UPSTAGE_API_KEY 입력
#   - all_data, kanana_solar_eval.jsonl 등 gitignore된 데이터 파일 로컬 PC에서 scp로 업로드
#     (all_data는 git에 안 올라가는 폴더라 매 pod마다 새로 올려야 함)
#
# 사용법:
#   git clone https://github.com/yewon4284/fairytale_llm.git && cd fairytale_llm
#   bash setup_pod.sh

set -e

echo "================================================================"
echo "  0/5. git 설정 (에디터 없어서 merge commit 막히는 문제 방지)"
echo "================================================================"
git config --global core.editor "true"
git config --global pull.rebase false
echo "  core.editor=true, pull.rebase=false 설정 완료 — 앞으로 git pull이 merge commit"
echo "  메시지 편집기를 찾다가 멈추는 일 없음"

echo ""
echo "================================================================"
echo "  1/5. requirements.txt 설치"
echo "================================================================"
pip install -r requirements.txt

echo ""
echo "================================================================"
echo "  2/5. GPU 드라이버에 맞는 torch/torchvision으로 재설치"
echo "================================================================"
DRIVER_CUDA=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
if [ -z "$DRIVER_CUDA" ]; then
    echo "  [경고] nvidia-smi로 드라이버 CUDA 버전을 못 읽었음. GPU가 없거나 드라이버 문제일 수 있음."
    echo "  torch 기본 설치(requirements.txt 버전)를 그대로 둠 — GPU 인식 안 되면 아래를 수동 확인:"
    echo "    nvidia-smi"
    echo "    python -c \"import torch; print(torch.cuda.is_available())\""
else
    echo "  드라이버 지원 CUDA 버전: $DRIVER_CUDA"
    # 드라이버가 지원하는 버전 이하의 가장 가까운 PyTorch 휠 태그 선택
    CUDA_MAJOR_MINOR=$(echo "$DRIVER_CUDA" | tr -d '.')
    if [ "$CUDA_MAJOR_MINOR" -ge 124 ]; then
        TORCH_TAG="cu124"
    elif [ "$CUDA_MAJOR_MINOR" -ge 121 ]; then
        TORCH_TAG="cu121"
    else
        TORCH_TAG="cu118"
    fi
    echo "  torch/torchvision을 ${TORCH_TAG} 빌드로 재설치..."
    pip uninstall -y torch torchvision torchaudio
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_TAG}"
    python -c "import torch, torchvision; print(f'  torch {torch.__version__} / torchvision {torchvision.__version__} / CUDA available: {torch.cuda.is_available()}')"
fi

echo ""
echo "================================================================"
echo "  3/5. HuggingFace 캐시를 큰 볼륨(/workspace)으로 이동"
echo "================================================================"
if [ -d "/workspace" ]; then
    rm -rf ~/.cache/huggingface
    mkdir -p /workspace/hf_cache
    export HF_HOME=/workspace/hf_cache
    if ! grep -q "HF_HOME" ~/.bashrc 2>/dev/null; then
        echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc
    fi
    echo "  HF_HOME=/workspace/hf_cache 로 설정 완료 (bashrc에도 등록)"
    df -h / /workspace
else
    echo "  [경고] /workspace 없음 — 이 pod엔 큰 영구 볼륨이 없는 것 같음. 모델 다운로드 시"
    echo "  디스크 부족 나면 df -h로 확인 후 대안 마운트 경로로 HF_HOME을 옮길 것."
fi

echo ""
echo "================================================================"
echo "  4/5. 남은 수동 작업 체크리스트"
echo "================================================================"
echo "  [ ] .env 파일에 UPSTAGE_API_KEY 입력됐는지 확인:"
echo "        cat .env"
echo "      없으면: echo 'UPSTAGE_API_KEY=실제키' > .env"
echo ""
echo "  [ ] Solar Pro3/Pro4/HCX-007 비교 실험 하려면 HCX_API_KEY도 추가:"
echo "        echo 'HCX_API_KEY=실제키' >> .env"
echo "      (console.clovastudio.ncloud.com에서 발급, 없으면 HCX 비교만 자동으로 건너뜀)"
echo ""
echo "  [ ] all_data 폴더 존재 확인 (441개여야 함, gitignore돼서 scp로 직접 올려야 함):"
echo "        ls all_data 2>/dev/null | wc -l"
echo "      없으면 로컬 PowerShell에서:"
echo "        & \"C:\\Windows\\System32\\OpenSSH\\scp.exe\" -P <포트> -i \$env:USERPROFILE\\.ssh\\<키> -r \\"
echo "            \"C:\\Users\\jyoun\\Desktop\\linux\\fairytale_llm\\all_data\" root@<IP>:~/fairytale_llm/"
echo ""
echo "  [ ] kanana_solar_eval.jsonl 존재 확인 (팀원 벤치마크 라벨, gitignore 아니지만 개인 업로드 파일이라 별도 확인):"
echo "        ls kanana_solar_eval.jsonl 2>/dev/null"
echo ""
echo "  [ ] finetune/kanana-s5-adapter/final_adapter에 adapter_model.safetensors 실제로 있는지"
echo "      (없으면 '베이스 모델로만 실행' 경고 뜸 — 정상 동작하지만 S5 강화 어댑터 없이 도는 것):"
echo "        ls finetune/kanana-s5-adapter/final_adapter/"
echo ""
echo "================================================================"
echo "  셋업 스크립트 완료. 위 체크리스트 확인 후 작업 시작하세요."
echo "================================================================"
