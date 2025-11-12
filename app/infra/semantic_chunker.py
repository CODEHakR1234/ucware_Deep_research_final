from __future__ import annotations
import re
from typing import List, Union, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.domain.page_element import PageElement
from app.domain.page_chunk   import PageChunk          # ★ NEW
from app.dto.summary_dto import SummaryRequestDTO

# ──────────────── 하이퍼 파라미터 ────────────────
_MAX, _OVF, _OVL = 1_500, 1_800, 200
sent_split = RecursiveCharacterTextSplitter(
    chunk_size=_MAX,
    chunk_overlap=_OVL,
    separators=[". ", "!\n", "?\n"],
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
    """

    def __init__(self, max_chunk_size: int = 1500, overflow_threshold: int = 1800, overlap: int = 200):
        self.max_chunk_size = max_chunk_size
        self.overflow_threshold = overflow_threshold
        self.overlap = overlap

    def group(
        self,
        els: List[PageElement],
        *,
        return_pagechunk: bool = False
    ) -> List[Union[str, PageChunk]]:
        print(f"[SemanticChunker] 청킹 시작: {len(els)}개 요소", flush=True)
        blocks, buf, figs = [], [], []

        def flush(page_no: int):
            """버퍼 내용을 하나의 청크로 밀어 넣는다."""
            if not buf:
                return

            joined = " ".join(buf).strip()
            texts  = (
                sent_split.split_text(joined)
                if len(joined) > self.overflow_threshold
                else [joined]
            )

            if return_pagechunk:
                blocks.extend(
                    PageChunk(page=page_no, text=t, figs=list(figs))
                    for t in texts
                )
            else:
                blocks.extend(texts)

            buf.clear()
            figs.clear()

        last_page = -1

        for el in els:
            # 페이지가 바뀌면 flush
            if el.page_no != last_page:
                flush(last_page)
                last_page = el.page_no

            if el.kind == "text":
                for p in _PAR_BR.split(el.content):
                    p = p.strip()
                    if not p:
                        continue
                    if _MD_HEADER.match(p):
                        flush(el.page_no)
                        buf.append(p)
                    elif _BULLET.match(p):
                        buf.append(p)
                    else:
                        buf.append(p)
            else:  # figure / table / graph
                # 이미지 정보만 수집 (플레이스홀더 교체는 하지 않음)
                # content가 bytes인 경우 처리
                content = el.content if isinstance(el.content, str) else "image_data"
                figs.append((el.id, content))

            if sum(len(x) for x in buf) > self.max_chunk_size + self.overlap:
                flush(el.page_no)

        flush(last_page)
        print(f"[SemanticChunker] 청킹 완료: {len(blocks)}개 청크", flush=True)
        return blocks

