#!/bin/bash
# run_vllm.sh: vLLM 서버 실행 스크립트 (Qwen3-30B-A3B-GPTQ-Int4, 포트 12000, expert parallel 사용)

# 1. 가상환경 활성화
echo "[1] 가상환경 활성화 중..."
source ~/vllm-data-storage/datastorage/bin/activate

# 2. Hugging Face 토큰 입력
echo ""
echo "[2] Hugging Face Access Token을 입력하세요"
read -p "🔑 HUGGING_FACE_HUB_TOKEN: " USER_TOKEN
echo ""
export HUGGING_FACE_HUB_TOKEN="$USER_TOKEN"

# 3. 캐시 디렉토리 환경변수 설정
export HF_HOME="/home/work/vllm-data-storage/huggingface-cache"
export VLLM_CACHE_ROOT="/home/work/vllm-data-storage/vllm-cache"

# 4. 포트 설정
PORT=12000

# 5. 모델 이름 고정
MODEL_NAME="Qwen/Qwen3-30B-A3B-GPTQ-Int4"
  
# 6. GPU 번호 입력
echo ""
read -p "🎯 사용할 GPU 번호를 입력하세요 (예: 0, 1, 2): " GPU_ID
if [ -z "$GPU_ID" ]; then
  GPU_ID=0
  echo "👉 기본값 GPU 0번 사용"
fi

echo ""
echo "[🚀] vLLM 서버를 모델 [$MODEL_NAME] 로 실행합니다 (포트: $PORT)"
echo "     🎯 GPU 번호: $GPU_ID"
echo "     ⏳ 모델 로딩에는 수십 초에서 수 분이 걸릴 수 있습니다."
echo "     백그라운드에서 실행되며 로그는 vllm.log 파일에 저장됩니다."

# 7. vLLM 서버 백그라운드 실행 (지정한 GPU에서만 실행)
CUDA_VISIBLE_DEVICES=$GPU_ID nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --enable_expert_parallel \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port $PORT > vllm.log 2>&1 &

# 8. 실행 확인 (최대 300초 대기)
MAX_WAIT=300
WAIT_TIME=0

echo ""
echo "[⌛] vLLM 서버가 포트 $PORT에서 열릴 때까지 대기 중... (최대 ${MAX_WAIT}s)"

while ! lsof -i :$PORT > /dev/null 2>&1; do
  sleep 2
  WAIT_TIME=$((WAIT_TIME + 2))
  if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo "❌ ${MAX_WAIT}초 내에 vLLM 서버가 시작되지 않았습니다. 로그(vllm.log)를 확인하세요."
    exit 1
  fi
done

VLLM_PID=$(lsof -i :$PORT -t)
echo ""
echo "✅ vLLM 서버가 정상적으로 실행 중입니다! (PID: $VLLM_PID)"
echo "📄 실시간 로그 보기: tail -f vllm.log"
