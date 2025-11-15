# app/infra/llm_engine.py
"""llm_engine.py
LLM 기반 Q&A 및 Map-Reduce 요약기.

갱신된 ``LlmChainIF`` 인터페이스를 구현::

    class LlmChainIF(Protocol):
        async def execute(self, prompt: str) -> str: ...
        async def summarize(self, docs: List[TextChunk]) -> str: ...

* **execute(prompt)** – 완전히 포맷된 프롬프트 문자열을 받아
  LLM 응답을 반환합니다.
* **summarize(docs)** – 주어진 텍스트 청크들에 대해
  LangChain의 ``map_reduce`` 방식으로 요약을 수행합니다.
"""


from __future__ import annotations

from typing import List, Dict, Any, Optional
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI

from app.infra.llm_factory import get_llm_instance
from app.domain.interfaces import LlmChainIF, TextChunk

# ───────────────────── 프롬프트 템플릿 정의 ─────────────────────

MAP_PROMPT = """
You are a helpful assistant that summarizes the following text.

{text}

Please summarize the text in a concise manner.

/no_think
"""

COMBINE_PROMPT = """
You are a helpful assistant that combines the following summaries.

{text}

Please combine the summaries in a concise manner.

/no_think
"""

# ───────────────────── LangChain 버그 패치 ─────────────────────

class PatchedChatOpenAI(ChatOpenAI):
    """vLLM 호환성을 위해 _combine_llm_outputs를 패치한 ChatOpenAI.
    
    vLLM의 OpenAI 호환 API가 토큰 사용량 정보를 제대로 반환하지 않아
    LangChain의 _combine_llm_outputs에서 None + None 오류가 발생하는 문제를 해결.
    """
    
    def _combine_llm_outputs(self, llm_outputs: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
        """여러 LLM 출력의 토큰 사용량을 안전하게 합산."""
        overall_token_usage: Dict[str, Any] = {}
        system_fingerprint = None
        
        for output in llm_outputs:
            if output is None:
                continue
                
            token_usage = output.get("token_usage", {})
            if token_usage:
                for k, v in token_usage.items():
                    if v is not None:  # None 체크 추가
                        # v가 딕셔너리인 경우 처리 (중첩된 token_usage 구조)
                        if isinstance(v, dict):
                            # 딕셔너리인 경우, 값을 합산할 수 없으므로 첫 번째 값을 사용하거나 합산
                            if k not in overall_token_usage:
                                overall_token_usage[k] = {}
                            if isinstance(overall_token_usage[k], dict):
                                # 중첩 딕셔너리 병합
                                for sub_k, sub_v in v.items():
                                    if isinstance(sub_v, (int, float)):
                                        if sub_k in overall_token_usage[k]:
                                            overall_token_usage[k][sub_k] += sub_v
                                        else:
                                            overall_token_usage[k][sub_k] = sub_v
                                continue
                            # overall_token_usage[k]가 정수인데 v가 딕셔너리면 v의 합계 사용
                            total = sum(sub_v for sub_v in v.values() if isinstance(sub_v, (int, float)))
                            overall_token_usage[k] = total
                        elif isinstance(v, (int, float)):
                            # v가 정수/실수인 경우 기존 로직
                            if k in overall_token_usage:
                                overall_token_usage[k] += v
                            else:
                                overall_token_usage[k] = v
            
            if system_fingerprint is None:
                system_fingerprint = output.get("system_fingerprint")
        
        combined = {"token_usage": overall_token_usage, "model_name": self.model_name}
        if system_fingerprint:
            combined["system_fingerprint"] = system_fingerprint
        
        return combined

# ───────────────────── LLM 엔진 구현체 ─────────────────────

class LlmEngine(LlmChainIF):
    """LlmChainIF 구현체로, LLM 기반 Q&A 및 요약 기능을 제공한다.

    Attributes:
        llm: LangChain LLM 실행 객체 (OpenAI 또는 HuggingFace 기반).
        map_prompt: 개별 문단 요약에 사용할 프롬프트 템플릿.
        combine_prompt: 통합 요약에 사용할 프롬프트 템플릿.
        _qa_chain: prompt → LLM → 출력 문자열 파서 체인.
        _summ_chain: 문서 리스트에 대한 map-reduce 요약 체인.
    """
    def __init__(self, *, temperature: float = 0.7):
        # Shared LLM instance
        self.llm = get_llm_instance(temperature=temperature)
        self.map_prompt = PromptTemplate(template=MAP_PROMPT, input_variables=["text"])
        self.combine_prompt = PromptTemplate(template=COMBINE_PROMPT, input_variables=["text"])

        # LangChain Runnable 체인을 구성: 입력 문자열 그대로 전달 → LLM 실행 → 문자열로 파싱
        self._qa_chain = (
            RunnablePassthrough()
            | self.llm
            | StrOutputParser()
        )

        # docs(list[str]) → map‑reduce → str (for *summarize*)
        # LangChain 0.2.x 호환: output_key 명시
        self._summ_chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            return_intermediate_steps=False,
            map_prompt=self.map_prompt,
            combine_prompt=self.combine_prompt,
            input_key="input_documents",
            output_key="output_text",
        )

    # ------------------------------------------------------------------
    # LlmChainIF implementation
    # ------------------------------------------------------------------
    async def execute(self, prompt: str, think: bool = False) -> str:  # noqa: D401
        """LLM call with a fully‑formatted *prompt* string.

        Args:
            prompt: 완성된 프롬프트 문자열.
            think: /no_think 명령어 생략 여부.

        Returns:
            LLM 응답 문자열 (후처리 포함).
        """
        if not think:
            prompt = prompt + "/no_think"
        result = (await self._qa_chain.ainvoke(prompt)).strip()
        # </think> 태그 제거: 시스템 메시지와 사용자 응답 분리 목적으로 삽입된 내용을 후처리로 제거
        if "</think>" in result:
            result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
        return result


    async def summarize(self, docs: List[TextChunk]) -> str:  # noqa: D401
        """High‑level summary using map‑reduce over *docs*.

        Args:
            docs: TextChunk 문자열 리스트.

        Returns:
            요약 문자열 결과값.
        """
        lc_docs = [Document(page_content=t) for t in docs]
        # ``ainvoke`` returns the final summary string when
        # ``return_intermediate_steps=False``.
        result = await self._summ_chain.ainvoke({"input_documents": lc_docs})

        # LangChain 버전에 따라 반환 타입이 다를 수 있음
        if isinstance(result, dict):
            # dict 형태: {"output_text": "..."}
            return str(result.get("output_text", "")).strip()
        elif isinstance(result, str):
            # 직접 문자열 반환
            return result.strip()
        else:
            # 알 수 없는 타입: 문자열 변환 시도
            return str(result).strip()