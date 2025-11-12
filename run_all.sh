#!/bin/bash

# 전체 실행 자동화 스크립트

echo "✅ Step 1: 가상환경 활성화"
#python -m venv .venv 
source .venv/bin/activate

#echo "✅ Step 1: 환경 설정 및 의존성 설치 중..."
#./scripts/setup_env.sh || { echo "❌ setup_env.sh 실패"; exit 1; }

echo "✅ Step 2: Redis 및 Chroma 서버 실행 중..."
./scripts/start_services.sh || { echo "❌ start_services.sh 실패"; exit 1; }

echo "✅ Step 3: OpenAI 키 등록 및 FastAPI 실행 중..."
./scripts/run_api.sh || { echo "❌ run_api.sh 실패"; exit 1; }

echo "🎉 모든 구성요소가 성공적으로 실행되었습니다!"

