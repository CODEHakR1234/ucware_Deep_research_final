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
        self.sim_threshold = 0.75  # 기존 0.78에서 완화 (너무 엄격하면 작은 그룹 증가)
        self.max_gap_pages = 2     # 기존 1에서 2로 확대 (관련 내용이 여러 페이지에 걸칠 수 있음)
        self.max_group_size = 5    # 기존 3에서 5로 증가 (더 풍부한 컨텍스트 제공)
        self.min_group_size = 2    # 최소 그룹 크기 (단일 청크 그룹 방지)
    
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
        
        # 후처리: 너무 작은 그룹 병합
        groups = self._merge_small_groups(groups)
        
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
        
        print(f"[SemanticGrouper] 그룹화 시작: {len(chunks)}개 청크", flush=True)
        print(f"[SemanticGrouper] 설정: threshold={self.sim_threshold}, max_gap={self.max_gap_pages}, max_size={self.max_group_size}", flush=True)
        
        for i in range(1, len(chunks)):
            chunk = chunks[i]
            embedding = embeddings[i]
            
            # 페이지 간격 계산
            gap = chunk.page - current_group[-1].page
            
            # 유사도 계산 (segment.py와 동일한 방식)
            sim = 1 - cosine(centroid, embedding)
            
            # 디버깅: 그룹화 결정 로그
            chunk_preview = chunk.text[:80] + "..." if len(chunk.text) > 80 else chunk.text
            decision_info = (
                f"청크 {i+1}(페이지 {chunk.page}): 유사도={sim:.3f}, "
                f"페이지간격={gap}, 그룹크기={len(current_group)}, "
                f"미리보기='{chunk_preview}'"
            )
            
            # 그룹화 조건 확인
            if (sim >= self.sim_threshold and 
                gap <= self.max_gap_pages and
                len(current_group) < self.max_group_size):
                # 같은 그룹에 추가
                print(f"[SemanticGrouper] ✅ {decision_info} → 그룹 {len(groups)+1}에 추가", flush=True)
                current_group.append(chunk)
                current_embeddings.append(embedding)
                # 중심벡터 업데이트 (전체 그룹의 평균으로 정확하게 계산)
                centroid = np.mean(current_embeddings, axis=0)
            else:
                # 새 그룹 시작
                reason = []
                if sim < self.sim_threshold:
                    reason.append(f"유사도낮음({sim:.3f}<{self.sim_threshold})")
                if gap > self.max_gap_pages:
                    reason.append(f"페이지간격큼({gap}>{self.max_gap_pages})")
                if len(current_group) >= self.max_group_size:
                    reason.append(f"그룹크기초과({len(current_group)}>={self.max_group_size})")
                
                print(f"[SemanticGrouper] 🔄 {decision_info} → 새 그룹 시작 (이유: {', '.join(reason)})", flush=True)
                groups.append(current_group)
                current_group = [chunk]
                current_embeddings = [embedding]
                centroid = embedding
        
        # 마지막 그룹 추가
        if current_group:
            groups.append(current_group)
        
        print(f"[SemanticGrouper] 초기 그룹화 완료: {len(groups)}개 그룹 생성", flush=True)
        for i, grp in enumerate(groups, 1):
            pages = [c.page for c in grp]
            print(f"[SemanticGrouper]   그룹 {i}: {len(grp)}개 청크, 페이지 {min(pages)}-{max(pages)}", flush=True)
        
        return groups
    
    def _merge_small_groups(self, groups: List[List[PageChunk]]) -> List[List[PageChunk]]:
        """
        너무 작은 그룹을 인접 그룹과 병합한다.
        단일 청크 그룹이나 min_group_size보다 작은 그룹을 처리한다.
        """
        if not groups or len(groups) == 1:
            return groups
        
        print(f"[SemanticGrouper] 소그룹 병합 시작: {len(groups)}개 그룹", flush=True)
        
        merged_groups = []
        i = 0
        
        while i < len(groups):
            current_group = groups[i]
            
            # 현재 그룹이 최소 크기 미만인 경우
            if len(current_group) < self.min_group_size:
                # 다음 그룹과 병합 시도
                if i + 1 < len(groups):
                    next_group = groups[i + 1]
                    # 병합해도 max_group_size를 초과하지 않는 경우에만 병합
                    if len(current_group) + len(next_group) <= self.max_group_size * 1.5:
                        merged = current_group + next_group
                        pages = [c.page for c in merged]
                        print(
                            f"[SemanticGrouper] 🔗 그룹 {i+1}({len(current_group)}개)와 "
                            f"그룹 {i+2}({len(next_group)}개) 병합 → "
                            f"{len(merged)}개 청크 (페이지 {min(pages)}-{max(pages)})",
                            flush=True
                        )
                        merged_groups.append(merged)
                        i += 2  # 두 그룹을 모두 처리했으므로 +2
                        continue
                
                # 병합할 수 없는 경우, 이전 그룹과 병합 시도
                if merged_groups and len(merged_groups[-1]) + len(current_group) <= self.max_group_size * 1.5:
                    prev_group = merged_groups[-1]
                    merged = prev_group + current_group
                    pages = [c.page for c in merged]
                    print(
                        f"[SemanticGrouper] 🔗 그룹 {i+1}({len(current_group)}개)을 "
                        f"이전 그룹({len(prev_group)}개)에 병합 → "
                        f"{len(merged)}개 청크 (페이지 {min(pages)}-{max(pages)})",
                        flush=True
                    )
                    merged_groups[-1] = merged
                else:
                    # 병합 불가능한 경우 그대로 추가 (마지막 그룹이거나 병합 시 너무 큰 경우)
                    print(
                        f"[SemanticGrouper] ⚠️ 그룹 {i+1}({len(current_group)}개)은 "
                        f"병합 불가능하여 그대로 유지",
                        flush=True
                    )
                    merged_groups.append(current_group)
            else:
                # 최소 크기를 만족하는 경우 그대로 추가
                merged_groups.append(current_group)
            
            i += 1
        
        print(f"[SemanticGrouper] 병합 완료: {len(merged_groups)}개 그룹 (병합 전: {len(groups)}개)", flush=True)
        for i, grp in enumerate(merged_groups, 1):
            pages = [c.page for c in grp]
            print(f"[SemanticGrouper]   최종 그룹 {i}: {len(grp)}개 청크, 페이지 {min(pages)}-{max(pages)}", flush=True)
        
        return merged_groups
    
    def set_similarity_threshold(self, threshold: float):
        """유사도 임계값을 설정한다."""
        self.sim_threshold = threshold
    
    def set_max_gap_pages(self, max_gap: int):
        """최대 페이지 간격을 설정한다."""
        self.max_gap_pages = max_gap
    
    def set_max_group_size(self, max_size: int):
        """최대 그룹 크기를 설정한다."""
        self.max_group_size = max_size
    
    def set_min_group_size(self, min_size: int):
        """최소 그룹 크기를 설정한다."""
        self.min_group_size = min_size
    
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