from __future__ import annotations
import re
from typing import List, Union, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.domain.page_element import PageElement
from app.domain.page_chunk   import PageChunk          # ★ NEW
from app.dto.summary_dto import SummaryRequestDTO

# ──────────────── 하이퍼 파라미터 ────────────────
# 튜토리얼 생성을 위해 청크 크기를 확대
# - 더 큰 청크 = 더 풍부한 컨텍스트 = 더 나은 설명 생성
_MAX, _OVF, _OVL = 2_000, 2_500, 300
sent_split = RecursiveCharacterTextSplitter(
    chunk_size=_MAX,
    chunk_overlap=_OVL,
    separators=["\n\n", ". ", "! ", "? ", "\n"],  # 분리자 우선순위 개선
)

_MD_HEADER = re.compile(r"^#{1,6}\s+.+")
# bullets 그대로 유지 (시장점 필요)
_BULLET = re.compile(r"^(\s*[\u2022\u2023\u25CF\-\*])|^\s*\d+\.\s+")
_PAR_BR = re.compile(r"\n{2,}")

# ─────────────────────────────────────────────────
class SemanticChunker:
    """
    PageElement 리스트 → semantic chunk 리스트로 변환.

    Parameters
    ----------
    return_pagechunk : True 이면 PageChunk 객체,
                       False 이면 plain str 을 돌려준다.
    
    개선 사항:
    - 페이지 경계 처리 완화 (작은 버퍼는 다음 페이지로 이어짐)
    - 마크다운 헤더 기반 청킹 개선
    - 이미지-텍스트 연관성 보존
    - 상세한 디버깅 로그
    """

    def __init__(self, max_chunk_size: int = 2000, overflow_threshold: int = 2500, overlap: int = 300):
        self.max_chunk_size = max_chunk_size
        self.overflow_threshold = overflow_threshold
        self.overlap = overlap
        self.min_chunk_size = 400  # 너무 작은 청크 방지 (기존 300에서 증가)

    def group(
        self,
        els: List[PageElement],
        *,
        return_pagechunk: bool = False
    ) -> List[Union[str, PageChunk]]:
        print(f"[SemanticChunker] 청킹 시작: {len(els)}개 요소", flush=True)
        blocks, buf, figs = [], [], []
        chunk_count = 0

        def flush(page_no: int, reason: str = ""):
            """버퍼 내용을 하나의 청크로 밀어 넣는다."""
            nonlocal chunk_count
            if not buf:
                return

            joined = " ".join(buf).strip()
            buf_size = len(joined)
            
            # 디버깅: flush 이유 출력
            if reason:
                print(f"[SemanticChunker] 🔄 Flush 트리거: {reason} (페이지 {page_no}, 버퍼 크기: {buf_size}자)", flush=True)
            
            # Overflow 처리: overflow_threshold를 초과하면 RecursiveCharacterTextSplitter로 분할
            texts  = (
                sent_split.split_text(joined)
                if buf_size > self.overflow_threshold
                else [joined]
            )
            
            # 디버깅: 분할 결과
            if len(texts) > 1:
                print(f"[SemanticChunker]   → Overflow로 {len(texts)}개 서브청크로 분할", flush=True)

            if return_pagechunk:
                for i, t in enumerate(texts):
                    chunk_count += 1
                    chunk = PageChunk(page=page_no, text=t, figs=list(figs))
                    blocks.append(chunk)
                    # 디버깅: 청크 생성 로그
                    img_ids = [img_id for img_id, _ in figs]
                    print(
                        f"[SemanticChunker]   ✅ 청크 {chunk_count} 생성: "
                        f"페이지 {page_no}, {len(t)}자, "
                        f"이미지 {len(img_ids)}개 {img_ids if img_ids else ''}",
                        flush=True
                    )
            else:
                blocks.extend(texts)
                chunk_count += len(texts)

            buf.clear()
            figs.clear()

        last_page = -1
        buf_has_header = False  # 버퍼에 헤더가 있는지 추적

        for idx, el in enumerate(els):
            current_buf_size = sum(len(x) for x in buf)
            
            # 페이지가 바뀌는 경우 처리 개선
            if el.page_no != last_page:
                # 버퍼가 충분히 크면 flush (의미 있는 청크)
                # 버퍼가 작으면 다음 페이지로 이어감 (문맥 보존)
                if current_buf_size > self.min_chunk_size:
                    flush(last_page, f"페이지 변경 ({last_page} → {el.page_no})")
                elif buf and current_buf_size > 0:
                    print(
                        f"[SemanticChunker] 📋 페이지 {last_page} → {el.page_no}: "
                        f"버퍼 {current_buf_size}자는 다음 페이지로 이어짐 (문맥 보존)",
                        flush=True
                    )
                last_page = el.page_no
                buf_has_header = False

            if el.kind == "text":
                for p in _PAR_BR.split(el.content):
                    p = p.strip()
                    if not p:
                        continue
                    
                    # 마크다운 헤더 처리 개선
                    if _MD_HEADER.match(p):
                        # 버퍼에 이미 내용이 있으면 flush (헤더는 새 청크의 시작)
                        if buf and not buf_has_header:
                            flush(el.page_no, f"마크다운 헤더 발견: '{p[:50]}...'")
                        buf.append(p)
                        buf_has_header = True
                    elif _BULLET.match(p):
                        buf.append(p)
                    else:
                        buf.append(p)
                        
            else:  # figure / table / graph
                # 이미지 플레이스홀더를 텍스트 버퍼에 추가
                buf.append(f"[{el.id}]")
                
                # 이미지 정보 수집
                content = el.content if isinstance(el.content, str) else "image_data"
                figs.append((el.id, content))
                
                print(f"[SemanticChunker] 🖼️ 이미지 추가: {el.id} (페이지 {el.page_no})", flush=True)

            # Overflow 체크 개선: overflow_threshold 사용
            current_buf_size = sum(len(x) for x in buf)
            if current_buf_size > self.overflow_threshold:
                flush(el.page_no, f"버퍼 오버플로우 ({current_buf_size} > {self.overflow_threshold})")
                buf_has_header = False

        # 마지막 버퍼 처리
        if buf:
            flush(last_page, "마지막 버퍼")
            
        print(f"[SemanticChunker] 청킹 완료: {len(blocks)}개 청크 생성", flush=True)
        return blocks

