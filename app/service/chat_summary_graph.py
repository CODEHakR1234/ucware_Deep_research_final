"""chat_summary_graph.py
채팅 요약·Q&A용 LangGraph 래퍼.

컨트롤러는 이 클래스를 통해 LangGraph 그래프에 접근하며, 그래프는 컴파일 타임에
한 번만 빌드해 싱글턴으로 재사용한다.
"""

from app.infra.llm_engine import LlmEngine
from .chat_graph_builder import ChatGraphBuilder, ChatState

# ────────────────────────── 그래프 싱글턴 빌드 ──────────────────────────
_builder_singleton = ChatGraphBuilder(llm=LlmEngine()).build()


class ChatSummaryGraph:
    """Chat LangGraph 래퍼.

    Attributes
    ----------
    graph : StateGraph
        사전에 컴파일된 LangGraph 싱글턴.
    """

    def __init__(self):
        self.graph = _builder_singleton  # 미리 빌드된 그래프

    async def generate(self, messages: list[str], query: str, lang: str):
        """그래프 실행 후 결과 딕셔너리를 반환한다.

        Args
        ----
        messages : list[str]
            정렬·포맷팅된 채팅 로그 문자열.
        query : str
            사용자 질문 문자열.
        lang : str
            번역 대상 언어(KO/EN).

        Returns
        -------
        dict
            summary/answer, log, error 등을 포함한 결과.
        """
        result = await self.graph.ainvoke(
            ChatState(messages=messages, query=query, lang=lang)
        )

        body = {"log": result.get("log", [])}
        if result.get("error"):
            body["error"] = result["error"]
            return body

        if result.get("is_summary"):
            body["summary"] = result.get("summary")
        else:
            body["answer"] = result.get("answer")
        return body


# ───────────────────── FastAPI Depends용 provider ─────────────────────
_service_singleton = ChatSummaryGraph()

def get_chat_summary_graph() -> ChatSummaryGraph:
    """FastAPI DI에서 사용될 싱글턴 반환 함수."""
    return _service_singleton

