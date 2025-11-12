from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4


@dataclass
class PageElement:
    """PDF 페이지 요소(텍스트·그림·표·그래프)."""
    kind: Literal["text", "figure", "table", "graph"]
    page_no: int
    content: str | bytes          # 텍스트 or 원본 이미지 bytes (이후 URL 로 치환)
    caption: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)  # 고유 ID

