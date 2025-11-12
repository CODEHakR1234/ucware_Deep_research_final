# app/controller/pdf_summary_controller.py

"""pdf_summary_controller.py
PDF 요약 생성 엔드포인트.

- URL: POST /api/summary
- 입력: SummaryRequestDTO  (file_id, pdf_url, query, lang)
- 출력: 요약 또는 질의 응답 결과(JSON)

실제 처리는 LangGraph 기반 서비스(`SummaryServiceGraph`)에 위임한다.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.dto.summary_dto import SummaryRequestDTO
from app.service.summary_service_graph import (
    SummaryServiceGraph,
    get_summary_service_graph,  # FastAPI Depends용
)

# ────────────────────────── Router 설정 ────────────────────────────────
router = APIRouter(prefix="/api")

@router.post("/summary", summary="PDF 요약 생성")
async def summarize_pdf(
    req: SummaryRequestDTO,
    service: SummaryServiceGraph = Depends(get_summary_service_graph),
):
    """PDF 요약 또는 질의 응답 결과를 반환한다.

    Args:
        req: SummaryRequestDTO – 파일 ID, PDF URL, 질문, 언어.
        service: LangGraph 기반 SummaryServiceGraph 인스턴스(FastAPI DI).

    Returns:
        dict: {file_id, summary|answer, cached, log, ...}

    Raises:
        HTTPException: 입력 검증 실패(ValueError) 시 400.
    """
    try:
        result = await service.generate(
            file_id=req.file_id,
            pdf_url=str(req.pdf_url),
            query=req.query,
            lang=req.lang,
        )
    except ValueError as e:
        # Service 층에서 검증 실패 시 400 에러로 매핑
        raise HTTPException(status_code=400, detail=str(e))

    return result  # 예: {"file_id": ..., "summary": ..., "cached": ...} 
