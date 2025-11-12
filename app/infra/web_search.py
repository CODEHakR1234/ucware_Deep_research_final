# app/infra/web_search.py
"""web_search.py
Tavily API를 활용한 웹 검색 구현체.

각 검색 결과를 개별적으로 분할해 TextChunk 단위로 반환한다.
"""

from langchain_community.tools.tavily_search.tool import TavilySearchResults
from app.domain.interfaces import WebSearchIF, TextChunk
import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


class WebSearch(WebSearchIF):
    """Tavily 기반 웹 검색 어댑터."""

    async def search(self, query: str, k: int = 5) -> List[TextChunk]:
        """웹 검색 결과를 텍스트 청크 단위로 반환한다.

        Args:
            query: 검색할 쿼리 문자열.
            k: 최대 검색 결과 수 (기본값 5).

        Returns:
            검색 결과 각 항목을 분할한 텍스트 청크 리스트.
        """
        # Tavily API 호출 (최대 k개의 결과 요청)
        web_search_tool = TavilySearchResults(
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            max_results=k
        )
        result = web_search_tool.run(query)
        
        # 검색 결과 각 항목을 개별적으로 분할
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )

        chunks: List[TextChunk] = []

        for item in result:
            if "content" not in item:
                continue

            content = item["content"]

            # 각 검색 결과에 대해 개별적으로 청크 분할
            docs = splitter.create_documents([content])

            # 텍스트만 추출하여 리스트에 추가
            chunks.extend(doc.page_content for doc in docs)
        print(chunks)
        return chunks