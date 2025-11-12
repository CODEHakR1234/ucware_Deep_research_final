# app/service/summary_service_graph.py
"""summary_service_graph.py
PDF 요약·질의 응답용 LangGraph 서비스를 감싸는 어댑터.

- 그래프는 **컴파일 타임**에 한 번만 빌드해 싱글턴으로 재사용한다.
- 컨트롤러는 이 클래스의 `generate()` 하나만 호출하면 된다.
"""

from app.infra.pdf_loader import PdfLoader
from app.infra.vector_store import VectorStore
from app.infra.llm_engine import LlmEngine
from app.infra.cache_store import CacheStore          
from app.infra.web_search import WebSearch
from app.domain.interfaces import CacheIF
from .summary_graph_builder import SummaryGraphBuilder, SummaryState

# ────────────────────────── 그래프 싱글턴 빌드 ──────────────────────────
_builder_singleton = SummaryGraphBuilder(
    PdfLoader(),
    VectorStore(),
    WebSearch(),
    LlmEngine(),
    CacheStore(),
)
_compiled_graph = _builder_singleton.build()


class SummaryServiceGraph:
    """컨트롤러 ↔ LangGraph 사이의 얇은 파사드."""

    def __init__(self):
        self.graph = _compiled_graph   # compiled graph shared

    # ───────────────────── Public API ─────────────────────
    async def generate(self, file_id: str, pdf_url: str, query: str, lang: str):
        """그래프를 실행해 요약 또는 답변 딕셔너리를 반환한다.

        Args
        ----
        file_id : str
            PDF 고유 ID.
        pdf_url : str
            PDF 다운로드 URL.
        query : str
            사용자 질문. "SUMMARY_ALL"이면 전체 요약 모드.
        lang : str
            답변 언어 코드(KO/EN).

        Returns
        -------
        dict
            호출자(컨트롤러)가 그대로 JSON 응답으로 내보낼 딕셔너리.
        """
        result = await self.graph.ainvoke(
            SummaryState(file_id=file_id, url=pdf_url, query=query, lang=lang),
            config={"recursion_limit": 100}
        )

        body = {
            "file_id": file_id,
            "cached": result.get("cached", False),
            "log" : result.get("log"),
        }
        if result.get("error"):
            body["error"] = result["error"]
            return body

        # `is_summary` is set by the EntryRouter in the graph
        if result.get("is_summary"):
            body["summary"] = result.get("answer")
        else:
            body["answer"] = result.get("answer")
        
        return body


# ───────────────────── FastAPI Depends 용 provider ─────────────────────
_service_singleton = SummaryServiceGraph()  

def get_summary_service_graph() -> SummaryServiceGraph:
    """FastAPI Depends용 싱글턴 반환"""
    return _service_singleton
