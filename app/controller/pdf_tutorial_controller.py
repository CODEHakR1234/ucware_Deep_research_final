from fastapi import APIRouter, Depends, HTTPException
from app.dto.tutorial_dto import TutorialRequestDTO
from app.service.guide_service_graph import get_guide_service, GuideServiceGraph

router = APIRouter(prefix="/api")

@router.post("/tutorial", summary="멀티모달 PDF 자습서 생성")
async def build_tutorial(
    req: TutorialRequestDTO,
    svc: GuideServiceGraph = Depends(get_guide_service),
):
    """튜토리얼을 생성한다."""
    try:
        result = await svc.generate(req.file_id, str(req.pdf_url), req.lang)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"튜토리얼 생성 중 오류가 발생했습니다: {str(e)}")

