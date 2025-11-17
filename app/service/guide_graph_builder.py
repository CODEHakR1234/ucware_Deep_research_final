# app/service/guide_graph_builder.py
"""GuideGraphBuilder – 멀티모달 PDF → 튜토리얼 Markdown 생성 파이프라인"""

from __future__ import annotations
import os
import re
import time
import asyncio
from functools import wraps
from typing import Awaitable, Callable, List, Optional

from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from app.domain.page_chunk import PageChunk
from app.domain.interfaces import (
    PdfLoaderIF,
    LlmChainIF,
    SemanticGrouperIF,
)
from app.prompts import (
    PROMPT_TUTORIAL,
    PROMPT_TUTORIAL_TRANSLATE,
    PROMPT_TUTORIAL_SECTION_WITH_IMAGES,
    PROMPT_TUTORIAL_SECTION_NO_IMAGES,
    PROMPT_TUTORIAL_TRANSLATE_WITH_IMAGES,
    PROMPT_TUTORIAL_TRANSLATE_NO_IMAGES,
)

# 상태 타입
class GuideState(BaseModel):
    """튜토리얼 생성 상태 모델."""
    file_id: str
    url: str
    lang: str
    
    chunks: List[PageChunk] = Field(default_factory=list)
    sections: List[str] = Field(default_factory=list)
    tutorial: Optional[str] = None
    
    cached: bool = False
    error: Optional[str] = None
    log: List[str] = Field(default_factory=list)

# Safe retry decorator
_RETRY = 3
_SLEEP = 1

def safe_retry(fn: Callable[[GuideState], Awaitable[GuideState]]):
    """노드에 재시도 로직을 적용하는 데코레이터."""
    @wraps(fn)
    async def _wrap(st: GuideState):
        for attempt in range(1, _RETRY + 1):
            try:
                t0 = time.perf_counter()
                result = await fn(st)
                elapsed = int((time.perf_counter() - t0) * 1000)
                st.log.append(f"{fn.__name__} attempt {attempt} [{elapsed}ms]")
                return result
            except Exception as exc:
                if attempt == _RETRY:
                    st.error = f"{fn.__name__} failed after {_RETRY} tries: {exc}"
                    return st
                await asyncio.sleep(_SLEEP)
        return st
    return _wrap

# Graph builder
class GuideGraphBuilder:
    """튜토리얼 생성 그래프 빌더."""

    def __init__(
        self,
        loader: PdfLoaderIF,
        grouper: SemanticGrouperIF,
        llm: LlmChainIF,
    ):
        self.loader = loader
        self.grouper = grouper
        self.llm = llm

    def _safe_truncate(self, text: str, max_len: int = 6000) -> str:
        """텍스트를 안전하게 자른다 (이미지 참조 보존)."""
        if len(text) <= max_len:
            return text
        
        truncated = text[:max_len]
        last_img_end = truncated.rfind(']')
        
        if last_img_end != -1:
            return truncated[:last_img_end + 1]
        
        return truncated

    def build(self) -> StateGraph:
        """튜토리얼 생성 그래프를 빌드한다."""
        g = StateGraph(GuideState)

        # 1. Load PDF
        @safe_retry
        async def load_pdf(st: GuideState):
            """PDF를 로드하여 청크로 변환한다."""
            st.chunks = await self.loader.load(st.url, with_figures=True)
            return st

        g.add_node("load", load_pdf)

        # 2. Generate sections
        @safe_retry
        async def generate_sections(st: GuideState):
            """자습서 섹션들을 생성한다 (병렬 처리)."""
            if not st.chunks:
                raise ValueError("PDF 청크가 없습니다.")
            
            # 청크 그룹화
            groups = self.grouper.group_chunks(st.chunks)
            st.log.append(f"청크 그룹화 완료: {len(groups)}개 그룹")
            
            # 병렬 섹션 생성
            async def _generate_one_section(grp, idx):
                try:
                    # 이 그룹에서 사용 가능한 이미지 추출
                    section_image_ids = set()
                    for chunk in grp:
                        for img_id, _ in chunk.figs:
                            section_image_ids.add(img_id)
                    
                    section_ids_str = ", ".join(sorted(section_image_ids)) if section_image_ids else "NONE"
                    chunks_text = self._safe_truncate("\n".join(c.text for c in grp), 6_000)
                    
                    # 프롬프트 생성
                    if section_image_ids:
                        prompt = PROMPT_TUTORIAL_SECTION_WITH_IMAGES.render(
                            available_image_ids=section_ids_str,
                            image_count=len(section_image_ids),
                            chunks=chunks_text
                        )
                    else:
                        prompt = PROMPT_TUTORIAL_SECTION_NO_IMAGES.render(
                            chunks=chunks_text
                        )
                    
                    section = await self.llm.execute(prompt)
                    return idx, section
                except Exception as e:
                    st.log.append(f"섹션 {idx} 생성 실패")
                    return idx, f"## 섹션 {idx}\n\n(생성 실패)"
            
            # 모든 그룹 병렬 처리
            tasks = [_generate_one_section(grp, i) for i, grp in enumerate(groups, 1)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 수집
            sections = []
            for result in sorted(results, key=lambda x: x[0] if isinstance(x, tuple) else 999):
                if isinstance(result, Exception):
                    sections.append("## 섹션\n\n(생성 실패)")
                elif isinstance(result, tuple):
                    idx, section = result
                    sections.append(section)
                    st.log.append(f"그룹 {idx} 섹션 생성 완료")

            st.sections = sections
            return st

        g.add_node("generate", generate_sections)

        # 3. Combine sections
        @safe_retry
        async def combine_sections(st: GuideState):
            """섹션들을 번역하고 하나의 Markdown 문서로 통합한다 (병렬 처리)."""
            if not st.sections:
                raise ValueError("섹션이 비어있습니다")
            
            valid_sections = [s for s in st.sections if s.strip()]
            if not valid_sections:
                raise ValueError("모든 섹션이 비어있습니다")
            
            # 병렬 번역
            async def _translate_one_section(section, idx):
                try:
                    # 섹션에 존재하는 이미지 ID 추출
                    section_img_ids = set(re.findall(r'IMG_\d+_\d+', section))
                    section_ids_str = ", ".join(sorted(section_img_ids)) if section_img_ids else "NONE"
                    
                    # 프롬프트 생성
                    if section_img_ids:
                        prompt = PROMPT_TUTORIAL_TRANSLATE_WITH_IMAGES.render(
                            image_count=len(section_img_ids),
                            available_image_ids=section_ids_str,
                            lang=st.lang,
                            text=section
                        )
                    else:
                        prompt = PROMPT_TUTORIAL_TRANSLATE_NO_IMAGES.render(
                            lang=st.lang,
                            text=section
                        )
                    
                    translated_section = await self.llm.execute(prompt)
                    return idx, translated_section
                except Exception as e:
                    st.log.append(f"섹션 {idx+1} 번역 실패 (원본 사용)")
                    return idx, section
            
            # 모든 섹션 병렬 번역
            tasks = [_translate_one_section(s, i) for i, s in enumerate(valid_sections)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 수집
            translated_sections = []
            for result in sorted(results, key=lambda x: x[0] if isinstance(x, tuple) else 999):
                if isinstance(result, Exception):
                    translated_sections.append("## 섹션\n\n(번역 실패)")
                elif isinstance(result, tuple):
                    idx, section = result
                    translated_sections.append(section)
            
            # 문서 구조로 결합
            st.tutorial = self._create_document_structure(translated_sections)
            return st

        g.add_node("combine", combine_sections)

        # Routing
        g.set_entry_point("load")

        def post_load(st: GuideState) -> str:
            return "finish" if st.error else "generate"

        g.add_conditional_edges("load", post_load, {
            "generate": "generate",
            "finish": "finish",
        })

        def post_generate(st: GuideState) -> str:
            return "finish" if st.error else "combine"

        g.add_conditional_edges("generate", post_generate, {
            "combine": "combine",
            "finish": "finish",
        })

        # combine → finish 직접 연결 (조건 불필요)
        g.add_edge("combine", "finish")

        # 4. Finish
        async def finish_node(st: GuideState):
            """이미지 ID를 base64 URI로 교체하고 종료한다."""
            if st.error:
                st.log.append(f"에러로 종료: {st.error}")
            else:
                # 이미지 ID를 base64 URI로 교체
                if st.tutorial and st.chunks:
                    image_mapping = self._create_image_mapping(st.chunks)
                    st.tutorial = self._replace_image_ids_with_uris(st.tutorial, image_mapping)
                    st.log.append(f"이미지 임베딩 완료: {len(image_mapping)}개")
                
                st.log.append("튜토리얼 생성 완료")
            return st

        g.add_node("finish", finish_node)

        g.set_finish_point("finish")
        return g.compile()

    # Helper methods
    def _create_image_mapping(self, chunks: List[PageChunk]) -> dict:
        """청크에서 이미지 ID를 URI로 매핑하는 딕셔너리 생성."""
        image_mapping = {}
        
        for chunk in chunks:
            for img_id, uri in chunk.figs:
                image_mapping[img_id] = uri
        
        return image_mapping

    def _replace_image_ids_with_uris(self, section: str, image_mapping: dict) -> str:
        """이미지 ID를 base64 URI로 교체."""
        # [IMG_X_Y] 또는 [IMG_X_Y:caption] 패턴 교체
        def replace_image_id(match):
            img_id = match.group(1)
            caption_text = match.group(2) if match.lastindex >= 2 else None
            
            if img_id in image_mapping:
                uri = image_mapping[img_id]
                alt_text = caption_text.strip()[:100] if caption_text else "이미지"
                return f"![{alt_text}]({uri})"
            else:
                return f"![이미지 로드 실패]() <!-- {img_id} 매핑 없음 -->"
        
        section_with_images = re.sub(
            r'\[(IMG_\d+_\d+)(?::([^\]]+))?\]', 
            replace_image_id, 
            section
        )
        
        # LLM이 변환한 ![...](IMG_X_Y) 패턴도 교체
        def replace_markdown_image(match):
            alt_text = match.group(1)
            img_id = match.group(2)
            
            if img_id in image_mapping:
                uri = image_mapping[img_id]
                return f"![{alt_text}]({uri})"
            else:
                return f"![{alt_text}]() <!-- {img_id} 매핑 없음 -->"
        
        section_with_images = re.sub(
            r'!\[([^\]]*)\]\((IMG_\d+_\d+)\)',
            replace_markdown_image,
            section_with_images
        )
        
        return section_with_images

    def _create_document_structure(self, sections: List[str]) -> str:
        """섹션들을 하나의 Markdown 문서로 통합한다 (Markdown 형식 강제)."""
        if not sections:
            return ""
        
        # Markdown 문서 시작
        combined = "# 자습서 가이드\n\n"
        combined += "---\n\n"
        
        # 각 섹션을 Markdown 형식으로 정리하여 추가
        for i, section in enumerate(sections, 1):
            if not section or not section.strip():
                continue
            
            # 섹션 정리: 불필요한 빈 줄 제거, 마지막 빈 줄 정리
            cleaned_section = section.strip()
            
            # 섹션이 헤딩으로 시작하지 않으면 추가
            if not cleaned_section.startswith('#'):
                cleaned_section = f"## 섹션 {i}\n\n{cleaned_section}"
            
            # 헤딩 뒤에 빈 줄이 없으면 추가 (Markdown 규칙)
            cleaned_section = re.sub(r'^(#{1,6} .+)\n([^\n])', r'\1\n\n\2', cleaned_section, flags=re.MULTILINE)
            
            # 리스트 앞뒤 빈 줄 확보 (Markdown 규칙)
            cleaned_section = re.sub(r'([^\n])\n([-*+] |\d+\. )', r'\1\n\n\2', cleaned_section)
            cleaned_section = re.sub(r'([-*+] .+|\d+\. .+)\n([^\n-*+\d\n])', r'\1\n\n\2', cleaned_section)
            
            # 코드 블록 앞뒤 빈 줄 확보
            cleaned_section = re.sub(r'([^\n`])\n(```)', r'\1\n\n\2', cleaned_section)
            cleaned_section = re.sub(r'(```)\n([^\n`])', r'\1\n\n\2', cleaned_section)
            
            # 섹션 추가 (섹션 간 구분자 포함)
            combined += cleaned_section + "\n\n---\n\n"
        
        # 마지막 불필요한 빈 줄 제거
        return combined.rstrip()

