# app/service/chat_graph_builder.py
"""chat_graph_builder.py
채팅 기록 요약·질문 응답용 LangGraph 빌더.

- `ChatState` : 그래프 실행 중 공유 상태
- `safe_retry` : I/O 노드 재시도 래퍼
- `ChatGraphBuilder` : 그래프 정의 및 컴파일

컨트롤러는 `ChatSummaryGraph` 래퍼를 통해 이 그래프를 호출한다.
"""

from __future__ import annotations
import asyncio
from functools import wraps
from typing import List, Optional, Awaitable, Callable
from langgraph.graph import StateGraph
from pydantic import BaseModel
from app.domain.interfaces import LlmChainIF, TextChunk

# ────────────────────────── 상태 모델 ────────────────────────────
class ChatState(BaseModel):
    """그래프 실행 중 사용되는 공유 상태."""

    messages: List[str]
    query: str
    lang: str = "ko"

    summary: Optional[str] = None
    answer:  Optional[str] = None
    is_summary: bool = False
    error: Optional[str] = None
    log: List[str] = []
    
    need_refine: bool = False

# ────────────────────────── 공통 설정 ────────────────────────────
_RETRY, _SLEEP = 3, 1

# ────────────────────────── 재시도 데코레이터 ───────────────────
def safe_retry(fn: Callable[[ChatState], Awaitable[ChatState]]):
    """노드 함수가 예외를 던질 때 최대 3회까지 재시도한다."""
    @wraps(fn)
    async def _wrap(st: ChatState):
        for i in range(1, _RETRY + 1):
            try:
                st.log.append(f"{fn.__name__}:{i}")
                return await fn(st)
            except Exception as e:
                if i == _RETRY:
                    st.error = f"{fn.__name__} failed: {e}"
                    return st
                await asyncio.sleep(_SLEEP)
    return _wrap

# ────────────────────────── 그래프 빌더 ───────────────────────────
class ChatGraphBuilder:
    def __init__(self, llm: LlmChainIF):
        self.llm = llm

    def build(self):
        g = StateGraph(ChatState)
        
        # entry --------------------------------------------------------
        async def entry(st: ChatState):
            st.is_summary = st.query.strip().upper() == "SUMMARY_ALL"
            return st
        g.add_node("entry", entry)

        # summarize ----------------------------------------------------
        @safe_retry
        async def summarize(st: ChatState):
            st.summary = await self.llm.summarize(st.messages)  # type: ignore[arg-type]
            return st
        g.add_node("summarize", summarize)

        # answer -------------------------------------------------------
        @safe_retry
        async def answer(st: ChatState):
            docs = "\n".join(st.messages)
            prompt = (
                "You are a helpful assistant. Using the following chat history, "
                "### Question:\n{query}\n\n"
                "### Chat history:\n{docs}\n\n"
                "### Answer:"
            ).format(query=st.query, docs=docs)

            st.answer = await self.llm.execute(prompt)
            return st
        # ⚠️ state key 'answer'와 충돌 방지를 위해 노드 이름만 변경
        g.add_node("answer_node", answer)
        
        # verify -------------------------------------------------------
        @safe_retry
        async def verify(st: ChatState):
            prompt = (
            "You are a helpful assistant that verifies if an answer is relevant to the chat history.\n\n"
            "Rules:\n"
            "- If the answer is unrelated to the chat history, return 'bad'\n"
            "- If the answer is partially incorrect or irrelevant, return 'false'\n"
            "- If the answer is correct and clearly based on the chat history, return 'true'\n"
            "- ONLY return one of: 'true', 'false', 'bad'\n\n"
            "### Question:\n{query}\n\n"
            "### Chat History:\n{docs}\n\n"
            "### Answer:\n{answer}\n\n"
            "### Verify:"
            ).format(query=st.query, docs="\n".join(st.messages), answer=st.answer)

            verdict = await self.llm.execute(prompt)
            st.log.append(f"verify:{verdict}")
            if "bad" in verdict.lower():
                st.answer = (
                    "I'm sorry, I don't know the answer to that question"
                    " because it's not related to the chat history. Please try again."
                )
                return st
            if "true" in verdict.lower():
                st.need_refine = False
            else:
                st.need_refine = True
            return st
        g.add_node("verify", verify)

        def post_verify(st: ChatState) -> str:
            if st.error:
                return "finish"
            return "refine" if st.need_refine else "translate"
        
        g.add_conditional_edges("verify", post_verify, {
            "refine": "refine", "translate": "translate", "finish": "finish"})

        # refine -------------------------------------------------------
        @safe_retry
        async def refine(st: ChatState):
            docs = "\n".join(st.messages)
            prompt = (
                "You are a helpful assistant. Using the following chat history, refine the answer."
                "### Question:\n{query}\n\n"
                "### Chat history:\n{docs}\n\n"
                "### Answer:\n{answer}\n\n"
                "### Refine:"
            ).format(query=st.query, docs=docs, answer=st.answer)

            st.answer = await self.llm.execute(prompt)
            return st
        g.add_node("refine", refine)
        
        def post_refine(st: ChatState) -> str:
            if st.error:
                return "finish"
            return "verify"
        
        g.add_conditional_edges("refine", post_refine, {
            "verify": "verify", "finish": "finish"})

        # translate ----------------------------------------------------
        async def translate(st: ChatState):
            text = st.summary if st.is_summary else st.answer

            prompt = (
            "You are a professional translator.\n"
            "Translate the following text **into {lang}**."
            "Preserve meaning and tone:\n\n"
            "{text}"
            ).format(lang=st.lang, text=text)

            translated = await self.llm.execute(prompt)

            if st.is_summary:
                st.summary = translated      
            else:
                st.answer  = translated
            return st
        
        g.add_node("translate", translate)

        # finish -------------------------------------------------------
        g.add_node("finish", lambda st: st)

        # ──────────── 그래프 구조 정의 ────────────
        g.set_entry_point("entry")
        g.add_conditional_edges(
            "entry",
            # 분기에서 'answer' → 'answer_node' 로 수정
            lambda st: "summarize" if st.is_summary else "answer_node",
            {"summarize": "summarize", "answer_node": "answer_node"},
        )
        g.add_edge("summarize", "translate")
        # 엣지에서도 'answer' → 'answer_node'
        g.add_edge("answer_node", "verify")
        g.add_edge("translate", "finish")

        g.set_finish_point("finish")
        return g.compile()