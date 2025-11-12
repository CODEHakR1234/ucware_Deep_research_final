from fastapi import APIRouter, Depends, HTTPException
from app.dto.tutorial_dto import TutorialRequestDTO, TutorialResponseDTO
from app.service.guide_service_graph import get_guide_service, GuideServiceGraph

router = APIRouter(prefix="/api")

@router.post("/tutorial", summary="멀티모달 PDF 자습서 생성", response_model=TutorialResponseDTO)
async def build_tutorial(
    req: TutorialRequestDTO,
    svc: GuideServiceGraph = Depends(get_guide_service),
):
    try:
        result = await svc.generate(req.file_id, str(req.pdf_url), req.lang)
        return TutorialResponseDTO(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[TutorialController] 예상치 못한 오류: {e}", flush=True)
        raise HTTPException(status_code=500, detail="튜토리얼 생성 중 오류가 발생했습니다.")

