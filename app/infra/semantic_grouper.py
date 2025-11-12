"""
semantic_grouper.py
------------------
기존 벡터DB 인프라를 활용하여 PageChunk들을 의미 단위로 그룹화하는 컴포넌트.
segment.py의 로직을 참고하되, PageChunk 특성에 맞게 최적화했다.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
from scipy.spatial.distance import cosine

from app.domain.page_chunk import PageChunk
from app.domain.interfaces import SemanticGrouperIF
from app.vectordb.vector_db import get_vector_db


class SemanticGrouper(SemanticGrouperIF):
    """벡터DB 기반 의미 단위 청크 그룹화기"""
    
    def __init__(self):
        """벡터DB 인스턴스를 초기화한다."""
        self.vdb = get_vector_db()
        self.sim_threshold = 0.78  # segment.py와 동일한 기본값
        self.max_gap_pages = 1     # segment.py와 동일한 기본값
        self.max_group_size = 3    # 튜토리얼용 추가 제한
    
    def group_chunks(self, chunks: List[PageChunk]) -> List[List[PageChunk]]:
        """
        PageChunk들을 의미 단위로 그룹화한다.
        
        Args:
            chunks: 그룹화할 PageChunk 리스트
            
        Returns:
            그룹화된 PageChunk 리스트들의 리스트
        """
        if not chunks:
            return []
        
        # 임베딩 생성 (벡터DB의 임베딩 모델 사용)
        embeddings = self._get_embeddings(chunks)
        
        # segment.py 방식으로 그룹화
        groups = self._group_by_similarity(chunks, embeddings)
        
        return groups
    
    def _get_embeddings(self, chunks: List[PageChunk]) -> List[np.ndarray]:
        """청크들의 임베딩을 생성한다."""
        if not chunks:
            return []
            
        texts = [chunk.text for chunk in chunks]
        
        # 벡터DB의 임베딩 모델 사용
        embeddings = []
        for i, text in enumerate(texts):
            try:
                # 벡터DB의 임베딩 함수 사용
                embedding = self.vdb.embeddings.embed_query(text)
                embeddings.append(np.array(embedding))
            except Exception as e:
                print(f"[SemanticGrouper] 청크 {i} 임베딩 생성 실패: {e}", flush=True)
                # 실패 시 0 벡터로 대체
                embeddings.append(np.zeros(384))  # 기본 차원
        
        return embeddings
    
    def _group_by_similarity(self, chunks: List[PageChunk], embeddings: List[np.ndarray]) -> List[List[PageChunk]]:
        """
        segment.py 방식을 참고하여 유사도 기반으로 청크들을 그룹화한다.
        """
        if not chunks:
            return []
        
        groups = []
        current_group = [chunks[0]]
        current_embeddings = [embeddings[0]]
        centroid = embeddings[0]  # 현재 그룹의 중심벡터
        
        for i in range(1, len(chunks)):
            chunk = chunks[i]
            embedding = embeddings[i]
            
            # 페이지 간격 계산
            gap = chunk.page - current_group[-1].page
            
            # 유사도 계산 (segment.py와 동일한 방식)
            sim = 1 - cosine(centroid, embedding)
            
            # 그룹화 조건 확인
            if (sim >= self.sim_threshold and 
                gap <= self.max_gap_pages and
                len(current_group) < self.max_group_size):
                # 같은 그룹에 추가
                current_group.append(chunk)
                current_embeddings.append(embedding)
                # 중심벡터 업데이트
                centroid = np.mean(np.vstack([centroid, embedding]), axis=0)
            else:
                # 새 그룹 시작
                groups.append(current_group)
                current_group = [chunk]
                current_embeddings = [embedding]
                centroid = embedding
        
        # 마지막 그룹 추가
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def set_similarity_threshold(self, threshold: float):
        """유사도 임계값을 설정한다."""
        self.sim_threshold = threshold
    
    def set_max_gap_pages(self, max_gap: int):
        """최대 페이지 간격을 설정한다."""
        self.max_gap_pages = max_gap
    
    def set_max_group_size(self, max_size: int):
        """최대 그룹 크기를 설정한다."""
        self.max_group_size = max_size
    
    def get_grouping_stats(self, chunks: List[PageChunk]) -> dict:
        """그룹화 통계를 반환한다."""
        if not chunks:
            return {"total_chunks": 0, "total_groups": 0, "avg_group_size": 0}
        
        groups = self.group_chunks(chunks)
        
        return {
            "total_chunks": len(chunks),
            "total_groups": len(groups),
            "avg_group_size": len(chunks) / len(groups) if groups else 0,
            "group_sizes": [len(group) for group in groups]
        }


# 싱글턴 인스턴스
_semantic_grouper = SemanticGrouper()

def get_semantic_grouper() -> SemanticGrouper:
    """SemanticGrouper 싱글턴을 반환한다."""
    return _semantic_grouper 