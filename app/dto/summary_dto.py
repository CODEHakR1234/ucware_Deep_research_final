"""summary_dto.py
PDF 요약·질의 응답 요청 DTO.
"""

from pydantic import BaseModel, HttpUrl

# ────────────────────────── 요청 DTO ────────────────────────────
class SummaryRequestDTO(BaseModel):
    """PDF 요약/QA 요청 바디."""

    file_id: str           # 문서 고유 ID
    pdf_url: HttpUrl       # PDF URL
    query: str             # 질문(또는 "SUMMARY_ALL")
    lang: str              # 답변 언어 코드(KO/EN)
