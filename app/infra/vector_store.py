# app/infra/vector_store.py
"""vector_store.py
LangChain‑Chroma 기반 벡터 스토어 어댑터.

문서 청크를 upsert·검색하는 역할만 담당하며, 상위 서비스는 **VectorStoreIF**만
의존하므로 다른 DB(Pinecone 등)로 교체하기 쉽다.
"""
from typing import List
from app.domain.interfaces import VectorStoreIF, TextChunk
from app.vectordb.vector_db import get_vector_db


class VectorStore(VectorStoreIF):
    """VectorStoreIF 구현체.

    Attributes
    ----------
    vdb : VectorDB
        Chroma 래퍼 싱글턴.
    """

    def __init__(self):
        self.vdb = get_vector_db()

    async def upsert(self, chunks: List[TextChunk], doc_id: str) -> None:
        """텍스트 청크를 벡터 DB에 저장한다."""
        self.vdb.store(chunks, doc_id)

    async def similarity_search(
        self, doc_id: str, query: str, k: int = 8
    ) -> List[TextChunk]:
        """벡터 유사도 검색 결과를 반환한다."""
        docs = self.vdb.get_docs(doc_id, query, k)
        return [d.page_content for d in docs]

    async def has_chunks(self, doc_id: str) -> bool:
        """해당 문서에 저장된 청크가 존재하는지 확인한다."""
        return self.vdb.has_chunks(doc_id)
    
    async def get_all(self, doc_id: str) -> List[TextChunk]:
        """문서의 모든 청크를 plain string 형태로 반환한다."""
        docs = self.vdb.get_all_chunks(doc_id)
        return [d.page_content for d in docs]
