#!/bin/bash
# stop_services.sh: uvicorn(FastAPI)

read -p "🌐 종료할 api포트 번호를 공백으로 구분해서 입력하세요 (예: 8000 8001 8002): " PORTS

for PORT in $PORTS; do
  echo "🛑 포트 $PORT에서 실행 중인 서버 종료 시도 중..."

  FASTAPI_PIDS=$(lsof -i :$PORT -t)

  if [ -n "$FASTAPI_PIDS" ]; then
    echo "$FASTAPI_PIDS" | xargs kill
    echo "✔ Fastapi 포트 $PORT의 프로세스(PID: $FASTAPI_PIDS) 종료 완료"
  else
    echo "⚠️  Fastapi 포트 $PORT에서 실행 중인 프로세스가 없습니다."
  fi
done
