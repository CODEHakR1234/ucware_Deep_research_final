# app/controller/feedback_controller.py

"""feedback_controller.py
클라이언트 **피드백 등록** 엔드포인트.

PDF 요약 결과에 대한 별점·코멘트를 Redis에 저장한 뒤
생성된 식별자와 타임스탬프를 반환한다.
"""

# 표준 라이브러리 -------------------------------------------------------
from datetime import datetime
from uuid import uuid4

# 서드파티 --------------------------------------------------------------
from fastapi import APIRouter, HTTPException, status

# 내부 모듈 --------------------------------------------------------------
from app.dto.feedback_dto import FeedbackCreate, FeedbackOut
from app.cache.cache_db import get_cache_db     

# ────────────────────────── Router 설정 ────────────────────────────────
router = APIRouter(prefix="/api")


@router.post(
    "/feedback",
    response_model=FeedbackOut,
    status_code=status.HTTP_201_CREATED,
    summary="PDF 요약 피드백 등록",
)
async def create_feedback(dto: FeedbackCreate) -> FeedbackOut:
    """피드백을 Redis에 저장하고 식별자를 반환한다.

    Args:
        dto: 파일 ID, 별점, 코멘트 등이 포함된 요청 바디.

    Returns:
        FeedbackOut: `id`, `created_at` 필드를 포함.

    Raises:
        HTTPException: Redis 작업 실패 시 500 반환.
    """
    try:
        feedback_id = str(uuid4())  # 피드백 고유 ID
        created_at  = datetime.utcnow()

        # Redis 저장 ---------------------------------------------------
        cache = get_cache_db()
        cache.add_feedback(                         
            file_id=dto.file_id,
            fb_id=feedback_id,
            payload={
                "file_id": dto.file_id,
                "pdf_url": str(dto.pdf_url),
                "lang": dto.lang,
                "rating": dto.rating,
                "comment": dto.comment,
                "usage_log": dto.usage_log,
                "created_at": created_at.isoformat(),
            },
        )

        return FeedbackOut(id=feedback_id, created_at=created_at)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"피드백 저장 실패: {exc}",
        )

