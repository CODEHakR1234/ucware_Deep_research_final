# app/controller/chat_summary_controller.py

"""chat_summary_controller.py
대화 기록 요약 / 질의 응답 엔드포인트.

- URL: POST /api/chat-summary
- 입력: ChatSummaryRequestDTO  (채팅 배열, 질문, 언어)
- 출력: 요약 또는 답변(JSON)

"""

from fastapi import APIRouter, Depends
from app.dto.chat_summary_dto import ChatSummaryRequestDTO
from app.service.chat_summary_graph import (
    ChatSummaryGraph,
    get_chat_summary_graph,
)

# ────────────────────────── Router 설정 ────────────────────────────────
router = APIRouter(prefix="/api")

@router.post("/chat-summary")
async def summarize_chat(
    req: ChatSummaryRequestDTO,
    service: ChatSummaryGraph = Depends(get_chat_summary_graph),
):
    """채팅 로그를 LangGraph에 전달해 요약/답변을 생성한다.

    Args:
        req: 채팅 배열·질문·언어가 담긴 요청 바디.
        service: DI로 주입되는 ChatSummaryGraph 싱글턴.

    Returns:
        dict: {"summary"|"answer", "log", ...}
    """
    # 채팅을 타임스탬프 기준 정렬한 뒤 한 줄 문자열로 변환
    msgs = sorted(req.chats, key=lambda c: c.timestamp)
    lines = [f"[{c.timestamp:%Y-%m-%d %H:%M:%S}] {c.sender}: {c.plaintext}"
             for c in msgs]

    return await service.generate(lines, query=req.query, lang=req.lang)

