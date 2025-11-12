from pydantic import BaseModel, HttpUrl
from typing import Optional, List

class TutorialRequestDTO(BaseModel):
    """Tutorial 생성 요청 DTO"""
    file_id: str
    pdf_url: HttpUrl
    lang: str = "ko"
    
    def __init__(self, **data):
        super().__init__(**data)
        # 지원하는 언어 검증
        supported_langs = ["ko", "en", "ja", "zh"]
        if self.lang not in supported_langs:
            raise ValueError(f"지원하지 않는 언어입니다: {self.lang}. 지원 언어: {supported_langs}")

class TutorialResponseDTO(BaseModel):
    """Tutorial 생성 응답 DTO"""
    file_id: str
    tutorial: Optional[str] = None
    cached: bool = False
    log: List[str] = []
    error: Optional[str] = None 