#!/bin/bash

# API 테스트용 curl 명령어 모음
# 사용법: ./test_curl.sh 또는 각 명령어를 개별적으로 실행

BASE_URL="http://localhost:8000"
# BASE_URL="http://192.168.0.173:8000"  # 필요시 변경

echo "=== API 테스트 curl 명령어 ==="
echo ""

# ────────────────────────────────────────────────────────────────
# 1. PDF 요약 생성 (POST /api/summary)
# ────────────────────────────────────────────────────────────────
echo "1. PDF 요약 생성:"
echo "curl -X POST ${BASE_URL}/api/summary \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{"
echo "    \"file_id\": \"fid_abc123_test\","
echo "    \"pdf_url\": \"https://arxiv.org/pdf/1706.03762.pdf\","
echo "    \"query\": \"SUMMARY_ALL\","
echo "    \"lang\": \"ko\""
echo "  }'"
echo ""

# 실제 실행 예시 (주석 해제하여 사용)
# curl -X POST ${BASE_URL}/api/summary \
#   -H 'Content-Type: application/json' \
#   -d '{
#     "file_id": "fid_abc123_test",
#     "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
#     "query": "SUMMARY_ALL",
#     "lang": "ko"
#   }'

# ────────────────────────────────────────────────────────────────
# 2. PDF 질의 응답 (POST /api/summary)
# ────────────────────────────────────────────────────────────────
echo "2. PDF 질의 응답:"
echo "curl -X POST ${BASE_URL}/api/summary \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{"
echo "    \"file_id\": \"fid_abc123_test\","
echo "    \"pdf_url\": \"https://arxiv.org/pdf/1706.03762.pdf\","
echo "    \"query\": \"이 문서의 주요 내용은 무엇인가요?\","
echo "    \"lang\": \"ko\""
echo "  }'"
echo ""

# ────────────────────────────────────────────────────────────────
# 3. 채팅 요약 생성 (POST /api/chat-summary)
# ────────────────────────────────────────────────────────────────
echo "3. 채팅 요약 생성:"
echo "curl -X POST ${BASE_URL}/api/chat-summary \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{"
echo "    \"chats\": ["
echo "      {"
echo "        \"chat_id\": \"msg_001\","
echo "        \"plaintext\": \"안녕하세요\","
echo "        \"sender\": \"user\","
echo "        \"timestamp\": \"2025-01-15T10:00:00\""
echo "      },"
echo "      {"
echo "        \"chat_id\": \"msg_002\","
echo "        \"plaintext\": \"네, 안녕하세요. 무엇을 도와드릴까요?\","
echo "        \"sender\": \"assistant\","
echo "        \"timestamp\": \"2025-01-15T10:00:05\""
echo "      }"
echo "    ],"
echo "    \"query\": \"SUMMARY_ALL\","
echo "    \"lang\": \"ko\""
echo "  }'"
echo ""

# ────────────────────────────────────────────────────────────────
# 4. 채팅 질의 응답 (POST /api/chat-summary)
# ────────────────────────────────────────────────────────────────
echo "4. 채팅 질의 응답:"
echo "curl -X POST ${BASE_URL}/api/chat-summary \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{"
echo "    \"chats\": ["
echo "      {"
echo "        \"chat_id\": \"msg_001\","
echo "        \"plaintext\": \"오늘 회의 일정이 어떻게 되나요?\","
echo "        \"sender\": \"user\","
echo "        \"timestamp\": \"2025-01-15T10:00:00\""
echo "      }"
echo "    ],"
echo "    \"query\": \"회의 일정을 알려주세요\","
echo "    \"lang\": \"ko\""
echo "  }'"
echo ""

# ────────────────────────────────────────────────────────────────
# 5. 피드백 등록 (POST /api/feedback)
# ────────────────────────────────────────────────────────────────
echo "5. 피드백 등록:"
echo "curl -X POST ${BASE_URL}/api/feedback \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{"
echo "    \"file_id\": \"fid_abc123_test\","
echo "    \"pdf_url\": \"https://arxiv.org/pdf/1706.03762.pdf\","
echo "    \"lang\": \"KO\","
echo "    \"rating\": 5,"
echo "    \"comment\": \"매우 유용한 요약이었습니다.\","
echo "    \"usage_log\": ["
echo "      \"첫 번째 질문: 문서 요약\","
echo "      \"두 번째 질문: 세부 내용 질의\""
echo "    ]"
echo "  }'"
echo ""

# ────────────────────────────────────────────────────────────────
# 6. PDF 튜토리얼 생성 (POST /api/tutorial)
# ────────────────────────────────────────────────────────────────
echo "6. PDF 튜토리얼 생성:"
echo "curl -X POST ${BASE_URL}/api/tutorial \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{"
echo "    \"file_id\": \"fid_abc123_test\","
echo "    \"pdf_url\": \"https://arxiv.org/pdf/1706.03762.pdf\","
echo "    \"lang\": \"ko\""
echo "  }'"
echo ""

# ────────────────────────────────────────────────────────────────
# 실제 실행 가능한 함수들
# ────────────────────────────────────────────────────────────────

test_pdf_summary() {
  echo "=== PDF 요약 테스트 ==="
  curl -X POST ${BASE_URL}/api/summary \
    -H 'Content-Type: application/json' \
    -d '{
      "file_id": "fid_abc123_test",
      "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
      "query": "SUMMARY_ALL",
      "lang": "ko"
    }' | jq .
  echo ""
}

test_pdf_qa() {
  echo "=== PDF 질의응답 테스트 ==="
  curl -X POST ${BASE_URL}/api/summary \
    -H 'Content-Type: application/json' \
    -d '{
      "file_id": "fid_abc123_qa_test",
      "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
      "query": "이 문서의 주요 내용은 무엇인가요?",
      "lang": "ko"
    }' | jq .
  echo ""
}

# ────────────────────────────────────────────────────────────────
# OpenAI API 키 확인 함수
# ────────────────────────────────────────────────────────────────
check_openai_key() {
  echo "=== OpenAI API 키 확인 ==="
  
  # .env 파일에서 API 키 읽기
  if [ -f .env ]; then
    API_KEY=$(grep "^OPENAI_API_KEY=" .env | cut -d'=' -f2 | tr -d '"' | tr -d "'")
    if [ -z "$API_KEY" ]; then
      echo "❌ .env 파일에서 OPENAI_API_KEY를 찾을 수 없습니다."
      return 1
    fi
    echo "✅ .env 파일에서 API 키를 찾았습니다 (길이: ${#API_KEY}자)"
    echo "   키 앞 10자: ${API_KEY:0:10}..."
  else
    echo "❌ .env 파일이 없습니다."
    read -p "API 키를 직접 입력하세요: " API_KEY
  fi
  
  echo ""
  echo "OpenAI API에 연결 테스트 중..."
  
  # OpenAI API 테스트
  RESPONSE=$(curl -s -w "\n%{http_code}" -X GET "https://api.openai.com/v1/models" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" 2>&1)
  
  HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
  BODY=$(echo "$RESPONSE" | sed '$d')
  
  if [ "$HTTP_CODE" == "200" ]; then
    echo "✅ API 키가 유효합니다!"
    echo "사용 가능한 모델 수: $(echo "$BODY" | jq '.data | length' 2>/dev/null || echo 'N/A')"
  elif [ "$HTTP_CODE" == "401" ]; then
    echo "❌ API 키가 유효하지 않거나 만료되었습니다."
    echo "응답: $BODY" | jq . 2>/dev/null || echo "$BODY"
    echo ""
    echo "💡 해결 방법:"
    echo "   1. OpenAI 대시보드에서 새 API 키 생성: https://platform.openai.com/api-keys"
    echo "   2. .env 파일의 OPENAI_API_KEY 업데이트"
    echo "   3. 서버 재시작"
  elif [ "$HTTP_CODE" == "429" ]; then
    echo "⚠️  API 사용량 한도 초과 (Rate Limit)"
    echo "응답: $BODY" | jq . 2>/dev/null || echo "$BODY"
  else
    echo "❌ API 연결 실패 (HTTP $HTTP_CODE)"
    echo "응답: $BODY" | jq . 2>/dev/null || echo "$BODY"
  fi
  echo ""
}

test_chat_summary() {
  echo "=== 채팅 요약 테스트 ==="
  curl -X POST ${BASE_URL}/api/chat-summary \
    -H 'Content-Type: application/json' \
    -d '{
      "chats": [
        {
          "chat_id": "msg_001",
          "plaintext": "안녕하세요",
          "sender": "user",
          "timestamp": "2025-01-15T10:00:00"
        },
        {
          "chat_id": "msg_002",
          "plaintext": "네, 안녕하세요. 무엇을 도와드릴까요?",
          "sender": "assistant",
          "timestamp": "2025-01-15T10:00:05"
        }
      ],
      "query": "SUMMARY_ALL",
      "lang": "ko"
    }' | jq .
  echo ""
}

test_feedback() {
  echo "=== 피드백 등록 테스트 ==="
  curl -X POST ${BASE_URL}/api/feedback \
    -H 'Content-Type: application/json' \
    -d '{
      "file_id": "fid_abc123_test",
      "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
      "lang": "KO",
      "rating": 5,
      "comment": "매우 유용한 요약이었습니다.",
      "usage_log": ["첫 번째 질문: 문서 요약"]
    }' | jq .
  echo ""
}

test_tutorial() {
  echo "=== PDF 튜토리얼 생성 테스트 ==="
  curl -X POST ${BASE_URL}/api/tutorial \
    -H 'Content-Type: application/json' \
    -d '{
      "file_id": "fid_abc123_test",
      "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
      "lang": "ko"
    }' | jq .
  echo ""
}

# 메인 실행
if [ "$1" == "summary" ]; then
  test_pdf_summary
elif [ "$1" == "qa" ] || [ "$1" == "question" ]; then
  test_pdf_qa
elif [ "$1" == "chat" ]; then
  test_chat_summary
elif [ "$1" == "feedback" ]; then
  test_feedback
elif [ "$1" == "tutorial" ]; then
  test_tutorial
elif [ "$1" == "check-key" ] || [ "$1" == "key" ]; then
  check_openai_key
elif [ "$1" == "all" ]; then
  test_pdf_summary
  test_pdf_qa
  test_chat_summary
  test_feedback
  test_tutorial
else
  echo "사용법:"
  echo "  ./test_curl.sh              # curl 명령어 예시 출력"
  echo "  ./test_curl.sh summary      # PDF 요약 테스트"
  echo "  ./test_curl.sh qa           # PDF 질의응답 테스트"
  echo "  ./test_curl.sh chat         # 채팅 요약 테스트"
  echo "  ./test_curl.sh feedback     # 피드백 등록 테스트"
  echo "  ./test_curl.sh tutorial     # 튜토리얼 생성 테스트"
  echo "  ./test_curl.sh all          # 모든 API 테스트"
  echo ""
  echo "  ./test_curl.sh check-key    # OpenAI API 키 유효성 확인 (추천!)"
  echo ""
  echo "⚠️  LLM 연결 문제가 있다면:"
  echo "  1. API 키 확인: ./test_curl.sh check-key"
  echo "  2. .env 파일 확인: cat .env | grep LLM_PROVIDER"
  echo "  3. LLM_PROVIDER=openai 일 때: OpenAI API Key 확인"
  echo "  4. LLM_PROVIDER=hf 일 때: vLLM 서버 실행 확인 (./run_vllm.sh)"
  echo "  5. 로그 확인: tail -f fastapi.log | grep -i error"
fi

