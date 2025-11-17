#!/bin/bash
# run_api.sh: OpenAI Key 설정 + Chroma 확인 + FastAPI 실행 (입력 포트 기반)

# ───────────────────────────────
# [0] 가상환경 활성화
# ───────────────────────────────
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "[0] 가상환경 활성화 완료"
else
    echo "❌ .venv 가상환경을 찾을 수 없습니다. 먼저 setup_env.sh를 실행해주세요."
    exit 1
fi

# ───────────────────────────────
# [1] 포트 번호 입력 받기 (기본값: 8000)
# ───────────────────────────────
read -p "🌐 사용할 포트 번호를 입력하세요 [기본값: 8000]: " PORT
PORT=${PORT:-8000}

# ───────────────────────────────
# [2] .env에서 OPENAI_API_KEY 등 환경변수 로딩
# ───────────────────────────────
if [ -f .env ]; then
  set -o allexport
  source .env
  set +o allexport
  echo "[ℹ️] .env의 환경 변수를 불러왔습니다"
else
  echo "❌ .env 파일이 없습니다. 먼저 setup_env.sh를 실행해주세요."
  exit 1
fi

# ───────────────────────────────
# [3] Chroma 서버 준비될 때까지 대기
# ───────────────────────────────
echo "[🕓] Chroma 서버 상태 확인 중..."

MAX_RETRIES=30
RETRY_INTERVAL=2
COUNTER=0

while ! curl -s http://localhost:9000 > /dev/null; do
  ((COUNTER++))
  echo "🔁 Chroma가 아직 준비되지 않음 ($COUNTER/$MAX_RETRIES). ${RETRY_INTERVAL}초 후 재시도..."
  if [ "$COUNTER" -ge "$MAX_RETRIES" ]; then
    echo "❌ Chroma 서버가 준비되지 않아 FastAPI를 실행할 수 없습니다."
    exit 1
  fi
  sleep $RETRY_INTERVAL
done

echo "[✅] Chroma 서버 연결 성공!"

# ───────────────────────────────
# [4] FastAPI 서버 실행 (백그라운드 + 로그)
# ───────────────────────────────
echo "[🚀] FastAPI 서버를 포트 $PORT 에서 실행 중..."
nohup uvicorn app.main:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 120 > fastapi.log 2>&1 &

sleep 5

# ───────────────────────────────
# [5] 서버 기동 확인
# ───────────────────────────────
if lsof -i :$PORT | grep LISTEN; then
  echo "✅ FastAPI 서버가 포트 $PORT 에서 정상적으로 실행되었습니다."
else
  echo "❌ FastAPI 서버가 포트 $PORT 에서 실행되지 않았습니다."
  exit 1
fi

echo "🎉 API 서버 실행 완료!"

