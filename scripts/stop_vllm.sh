#!/bin/bash
# stop_vllm.sh: 포트 12000에서 실행 중인 vLLM 서버 종료 스크립트

PORT=12000

echo "[🛑] vLLM 서버 종료 시도 중... (포트: $PORT)"
VLLM_PID=$(lsof -i :$PORT -t)

if [ -n "$VLLM_PID" ]; then
  kill "$VLLM_PID"
  echo "✅ vLLM 서버(PID $VLLM_PID)가 정상적으로 종료되었습니다."
else
  echo "⚠️  포트 $PORT에서 실행 중인 vLLM 서버가 없습니다."
fi

