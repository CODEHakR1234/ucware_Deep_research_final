# app/vectordb/vector_db.py
"""vector_db.py
Chroma-based 벡터 DB 래퍼.

PDF 텍스트 청크를 임베딩 후 저장·검색하며, 삭제/모니터링 유틸리티도 포함한다.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime
from functools import lru_cache
from typing import List, Union

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from zoneinfo import ZoneInfo

from app.cache.cache_db import get_cache_db   # 삭제 로그용

# ─────────────────────── 설정 상수 ──────────────────────────
CHUNK_SIZE      = 3000
CHUNK_OVERLAP   = 300
_BATCH_SIZE     = 500

CHROMA_HOST     = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT     = int(os.getenv("CHROMA_PORT", "9000"))

LLM_PROVIDER        = os.getenv("LLM_PROVIDER", "openai")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

_PERSIST_DIR    = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# ───────────────── Embedding 모델 선택 ────────────────────
def _get_embedding_model():
    """환경 변수 설정에 따라 임베딩 모델(OpenAI/HF)을 반환한다."""
    if LLM_PROVIDER.lower() == "hf":
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )
    return OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

# ───────────────────────── VectorDB ─────────────────────────
class VectorDB:
    """Chroma 벡터 스토어 헬퍼.

    Notes
    -----
    - Chroma 클라이언트는 lazy 연결로 성능 이슈를 줄인다.
    - `store`에서 청크 분할·임베딩·문서 저장을 한 번에 처리한다.
    """

    def __init__(self) -> None:
        self.embeddings = _get_embedding_model()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

        self._lock   = threading.RLock()
        self._client = None                       

    # ──────────── Chroma client (lazy) ────────────
    @property
    def client(self) -> chromadb.HttpClient | None:
        if self._client is None:
            try:
                print(f"[VectorDB] Connecting → {CHROMA_HOST}:{CHROMA_PORT}")
                self._client = chromadb.HttpClient(
                    host=CHROMA_HOST,
                    port=CHROMA_PORT,
                )
                print("[VectorDB] ✅ Chroma connection OK")
            except Exception as e:
                print(f"[VectorDB] ❌ Chroma connect failed: {e}")
                self._client = None
        return self._client

    # ──────────── 내부 헬퍼 ────────────
    def _get_collection_name(self, file_id: str) -> str:
        return file_id

    def _get_vectorstore(self, file_id_or_col: str) -> Chroma:
        """Chroma 컬렉션 객체를 반환한다."""
        if self.client is None:
            raise RuntimeError("Chroma client not available")
        return Chroma(
            client=self.client,
            collection_name=file_id_or_col,
            embedding_function=self.embeddings,
            persist_directory=_PERSIST_DIR,
        )

    # ------------- CRUD 메서드 ----------------------------
    def store(self, content: Union[str, List[str]], file_id: str) -> None:
        """텍스트(또는 청크 리스트)를 임베딩 후 저장한다."""
        try:
            chunks = (
                self.text_splitter.split_text(content)
                if isinstance(content, str)
                else content
            )
            if not chunks:
                print(f"[VectorDB.store] ⚠️ no chunks for {file_id}")
                return

            today = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
            docs: List[Document] = [
                Document(
                    page_content=ck,
                    metadata={
                        "file_id": file_id,
                        "chunk_index": idx,
                        "date": today,
                        "_type": "Document",  # ChromaDB 호환성을 위한 타입 정보
                    },
                )
                for idx, ck in enumerate(chunks)
            ]

            vs = self._get_vectorstore(self._get_collection_name(file_id))
            with self._lock:
                for i in range(0, len(docs), _BATCH_SIZE):
                    try:
                        vs.add_documents(docs[i : i + _BATCH_SIZE])
                    except Exception as e:
                        print(f"[VectorDB.store] batch {i//_BATCH_SIZE} fail: {e}")

            print(f"[VectorDB.store] ✅ stored {len(docs)} docs for {file_id}")

        except Exception as e:
            print(f"[VectorDB.store] ❌ {e}")

    def get_docs(self, file_id: str, query: str, k: int = 8) -> List[Document]:
        """유사도 검색 결과를 반환한다."""
        try:
            return self._get_vectorstore(self._get_collection_name(file_id)).similarity_search(query, k=k)
        except Exception as e:
            print(f"[VectorDB.get_docs] ❌ {e}")
            return []

    def get_all_chunks(self, file_id: str) -> List[Document]:
        """저장된 모든 청크를 chunk_index 순으로 반환한다."""
        try:
            col = self.client.get_collection(self._get_collection_name(file_id))  # type: ignore
            data = col.get(include=["documents", "metadatas"])
            docs_raw  = data.get("documents", [])
            metas_raw = data.get("metadatas", [{}] * len(docs_raw))

            items = sorted(
                zip(docs_raw, metas_raw),
                key=lambda x: x[1].get("chunk_index", 0),
            )
            return [Document(page_content=d, metadata=m) for d, m in items]
        except Exception as e:
            print(f"[VectorDB.get_all_chunks] ❌ {e}")
            return []

    def has_chunks(self, file_id: str) -> bool:
        """해당 파일에 저장된 청크가 하나라도 있는지 확인."""
        try:
            return self.client.get_collection(self._get_collection_name(file_id)).count() > 0  # type: ignore
        except Exception:
            return False

    def delete_document(self, file_id: str) -> bool:
        """컬렉션 전체를 삭제하고 로그를 남긴다."""
        try:
            with self._lock:
                self.client.delete_collection(self._get_collection_name(file_id))  # type: ignore
            self._log_vector_deletion(file_id)
            return True
        except Exception as e:
            print(f"[VectorDB.delete_document] ❌ {e}")
            return False

    def is_chroma_alive(self) -> bool:
        """Chroma 서버 헬스 체크."""
        try:
            self.client.heartbeat() 
            return True
        except Exception:
            return False


# 싱글턴 getter ---------------------------------------------------------
@lru_cache(maxsize=1)
def get_vector_db() -> "VectorDB":
    return VectorDB()

