"""chat_summary_dto.py
채팅 기록 요약·질의 응답 요청용 DTO 정의.

- **ChatMessageDTO** : 단일 메시지 구조
- **ChatSummaryRequestDTO** : 메시지 배열 + 질문 + 언어 코드
"""

from typing import List
from pydantic import BaseModel
from datetime import datetime

# ────────────────────────── 개별 메시지 ────────────────────────────
class ChatMessageDTO(BaseModel):
    """채팅 메시지 스키마."""

    chat_id: str
    plaintext: str
    sender: str
    timestamp: datetime

# ────────────────────────── 요청 바디 ────────────────────────────
class ChatSummaryRequestDTO(BaseModel):
    """채팅 요약/응답 요청 스키마."""

    chats: List[ChatMessageDTO]
    query: str
    lang: str = "ko"
