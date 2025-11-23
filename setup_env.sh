#!/bin/bash
set -e

# 임시 디렉터리를 /home/work/skku/tmp로 설정하여 디스크 공간 문제 해결

#TMP_DIR="/home/work/skku/tmp"
#mkdir -p "$TMP_DIR"
#export TMPDIR="$TMP_DIR"
#export TMP="$TMP_DIR"
#export TEMP="$TMP_DIR"


echo "[1] 시스템 패키지 설치"
sudo apt update
sudo apt install -y redis-server libgl1 libglib2.0-0

echo "[2] Python 가상환경 생성 및 활성화"
python3 -m venv .venv
source .venv/bin/activate

echo "[3] pip 패키지 설치 (공통)"
pip install --upgrade pip
pip install -r requirements.txt   # FastAPI/LangChain/Docling 등
echo "[3-1] ColPali/vision deps 설치"
pip install "colpali-engine>=0.3.1" "transformers>4.45.0" pillow PyMuPDF httpx chromadb
echo "[3-2] PyTorch (CUDA 12.1) 설치"
pip install torch --index-url https://download.pytorch.org/whl/cu121

# ──────────────── LLM / Embedding Provider 선택 ────────────────
echo ""
echo "🤖 사용할 LLM/Embedding Provider를 선택하세요:"
echo "1. openai"
echo "2. hf (HuggingFace)"
read -p "선택 [1/2]: " PROVIDER_CHOICE

echo -n "🔑 Tavily API Key를 입력하세요: "
read -r TAVILY_API_KEY

echo -n "🎮 사용할 GPU 번호를 입력하세요 (여러 장이면 콤마, 기본 0): "
read -r GPU_NUMBER
GPU_NUMBER=${GPU_NUMBER:-0}

# ──────────────── Hugging Face 캐시 경로 설정 ────────────────
echo ""
echo -n "📦 Hugging Face 캐시 디렉토리 경로를 입력하세요 (기본: /home/work/vllm-data-storage/huggingface_cache/huggingface): "
read -r HF_CACHE_DIR
HF_CACHE_DIR=${HF_CACHE_DIR:-/home/work/vllm-data-storage/huggingface_cache/huggingface}

# 캐시 디렉토리 생성
mkdir -p "$HF_CACHE_DIR"
echo "✅ Hugging Face 캐시 디렉토리: $HF_CACHE_DIR"

# ──────────────── 캡셔닝 백엔드 선택 ────────────────
echo ""
echo "🖼️ 사용할 캡셔닝 백엔드를 선택하세요:"
echo "1. vLLM(OpenAI 호환, LLaVA 등)"
echo "2. OpenAI 클라우드(멀티모달)"
echo "3. 사용 안함"
read -p "선택 [1/2/3]: " CAPTION_CHOICE

# 기본값: vLLM(LLaVA)
CAPTION_BACKEND="openai_local"
CAPTION_API_BASE="http://localhost:12001/v1"
CAPTION_OPENAI_MODEL="llava-hf/llava-1.6-7b-hf"
DISABLE_CAPTIONING=false

if [ "$CAPTION_CHOICE" == "2" ]; then
    CAPTION_BACKEND="openai"
    if [ -z "${OPENAI_API_KEY}" ]; then
        echo -n "🔑 OpenAI API Key(캡셔닝용)를 입력하세요: "
        read -r OPENAI_API_KEY
    fi
    CAPTION_API_BASE="https://api.openai.com/v1"
    echo -n "🧠 OpenAI 멀티모달 모델명(gpt-4o-mini 기본): "
    read -r CAPTION_OPENAI_MODEL_IN
    CAPTION_OPENAI_MODEL=${CAPTION_OPENAI_MODEL_IN:-gpt-4o-mini}
elif [ "$CAPTION_CHOICE" == "3" ]; then
    DISABLE_CAPTIONING=true
    CAPTION_BACKEND=""
    CAPTION_API_BASE=""
    CAPTION_OPENAI_MODEL=""
else
    # 선택 1: vLLM(LLaVA) 커스터마이즈 입력
    echo -n "🌐 캡셔너 API Base(http://localhost:12001/v1 기본): "
    read -r CAPTION_API_BASE_IN
    CAPTION_API_BASE=${CAPTION_API_BASE_IN:-http://localhost:12001/v1}
    echo -n "🧠 LLaVA 모델명(llava-hf/llava-1.6-7b-hf 기본): "
    read -r CAPTION_MODEL_IN
    CAPTION_OPENAI_MODEL=${CAPTION_MODEL_IN:-llava-hf/llava-1.6-7b-hf}
fi

if [ "$PROVIDER_CHOICE" == "2" ]; then
    LLM_PROVIDER="hf"
    EMBEDDING_MODEL_NAME="BAAI/bge-m3"
    LLM_MODEL_NAME="Qwen/Qwen3-30B-A3B-GPTQ-Int4"
else
    LLM_PROVIDER="openai"
    EMBEDDING_MODEL_NAME="text-embedding-ada-002"
    LLM_MODEL_NAME="gpt-3.5-turbo"
    echo -n "🔑 OpenAI API Key를 입력하세요: "
    read -r OPENAI_API_KEY
fi

# ──────────────── .env 생성 ────────────────
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
# ───────── 캡셔닝 설정 ─────────
CAPTION_BACKEND=$CAPTION_BACKEND
CAPTION_API_BASE=$CAPTION_API_BASE
CAPTION_OPENAI_MODEL=$CAPTION_OPENAI_MODEL
DISABLE_CAPTIONING=$DISABLE_CAPTIONING
# ───────── Hugging Face 캐시 설정 ─────────
HF_HOME=$HF_CACHE_DIR
TRANSFORMERS_CACHE=$HF_CACHE_DIR
HF_HUB_CACHE=$HF_CACHE_DIR/hub
EOF

echo "[✔] 환경 구성이 완료되었습니다."
echo "가상환경 활성화:  source .venv/bin/activate"

