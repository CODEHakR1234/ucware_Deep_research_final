#!/bin/bash
# setup_env.sh: Python 패키지 설치 + .env 생성

echo "[1] 시스템 패키지 설치"
sudo apt update
sudo apt install -y redis-server

echo "[2] Python 가상환경 생성 및 활성화"
python3 -m venv .venv
source .venv/bin/activate

echo "[3] pip 패키지 설치"
pip install --upgrade pip
pip install -r requirements.txt


echo "🤖 사용할 LLM/Embedding Provider를 선택하세요:"
echo "1. openai"
echo "2. hf (HuggingFace)"
read -p "선택 [1/2]: " PROVIDER_CHOICE

echo "🔑 Tavily API Key를 입력하세요: "
read -r TAVILY_API_KEY

if [ "$PROVIDER_CHOICE" == "2" ]; then
    LLM_PROVIDER="hf"
    EMBEDDING_MODEL_NAME="BAAI/bge-m3"
    LLM_MODEL_NAME="Qwen/Qwen3-30B-A3B-GPTQ-Int4"
    OPENAI_API_KEY=""
    GPU_NUMBER=0
    echo -n "사용할 GPU 번호를 입력하세요: (기본값 0)"
    read -r GPU_NUMBER
else
    LLM_PROVIDER="openai"
    EMBEDDING_MODEL_NAME="text-embedding-ada-002"
    LLM_MODEL_NAME="gpt-3.5-turbo"

    echo -n "🔑 OpenAI API Key를 입력하세요: "
    read -r OPENAI_API_KEY
fi

echo "[4] .env 파일 생성"
cat > .env <<EOF
CHROMA_HOST=localhost
CHROMA_PORT=9000
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_TTL=604800
LLM_PROVIDER=$LLM_PROVIDER
EMBEDDING_MODEL_NAME=$EMBEDDING_MODEL_NAME
LLM_MODEL_NAME=$LLM_MODEL_NAME
OPENAI_API_KEY="$OPENAI_API_KEY"
TAVILY_API_KEY="$TAVILY_API_KEY"
CUDA_VISIBLE_DEVICES=$GPU_NUMBER
EOF

echo "[✔] .env 생성 완료"

