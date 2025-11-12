# app/infra/embedder.py
from typing import Callable
import numpy as np

from app.vectordb.vector_db import get_vector_db


def get_embed_fn() -> Callable[[str], np.ndarray]:
    """
    외부 모듈이 임베딩 모델 세부 구현을 몰라도 되도록
    (str) -> np.ndarray  형태의 함수를 리턴한다.
    """
    vdb = get_vector_db()                # Chroma + Embeddings 싱글턴
    return lambda text: vdb.embeddings.embed_query(text)

