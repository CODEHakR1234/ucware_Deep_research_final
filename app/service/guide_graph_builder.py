# app/service/guide_graph_builder.py
"""GuideGraphBuilder – 멀티모달 PDF → 튜토리얼 Markdown
도메인 기반 LangGraph 파이프라인 빌더.
모든 노드는 최대 3회 재시도하며, 실패 시 에러 메시지를 기록하고 종료한다.
"""

from __future__ import annotations
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
from app.prompts import PROMPT_TUTORIAL, PROMPT_TUTORIAL_TRANSLATE

# ──────────────── 상태 타입 ────────────────
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

# ──────────────── Helper: safe-retry decorator ────────────────
_RETRY = 3
_SLEEP = 1  # seconds between retries

def safe_retry(fn: Callable[[GuideState], Awaitable[GuideState]]):
    """LangGraph 노드에 재시도 로직을 적용하는 데코레이터.

    `_RETRY` 횟수만큼 재시도하며 마지막 실패 시 상태 객체에 에러를 기록한다.

    Args:
        fn: GuideState를 받아 비동기로 처리하는 함수이자 노드.

    Returns:
        자동 재시도 로직이 적용된 비동기 함수.
    """
    @wraps(fn)
    async def _wrap(st: GuideState):  # type: ignore[override]
        for attempt in range(1, _RETRY + 1):
            try:
                t0 = time.perf_counter()
                result = await fn(st)
                elapsed = int((time.perf_counter() - t0) * 1000)  # ms
                st.log.append(
                    f"{fn.__name__} attempt {attempt} [{elapsed}ms]"
                )
                return result
            except Exception as exc:  # noqa: BLE001
                if attempt == _RETRY:
                    st.error = f"{fn.__name__} failed after {_RETRY} tries: {exc}"
                    return st
                await asyncio.sleep(_SLEEP)
        return st  # nothing should reach here

    return _wrap

# ──────────────── Graph builder ────────────────
class GuideGraphBuilder:
    """튜토리얼 생성 그래프를 빌드하는 LangGraph 파이프라인 생성기.

    Attributes:
        loader: PDF 로더 (URL → PageChunk 리스트).
        grouper: 의미 단위 청크 그룹화기.
        llm: LangChain 호환 LLM 실행기.
    """

    def __init__(
        self,
        loader: PdfLoaderIF,
        grouper: SemanticGrouperIF,
        llm: LlmChainIF,
    ):
        self.loader = loader
        self.grouper = grouper
        self.llm = llm

    def build(self) -> StateGraph:
        """튜토리얼 생성 그래프를 빌드한다."""
        g = StateGraph(GuideState)

        # 1. Load PDF ---------------------------------------------------
        @safe_retry
        async def load_pdf(st: GuideState):
            """PDF를 로드하여 청크로 변환한다.

            Args:
                st: 현재 요청 상태.

            Returns:
                텍스트 청크가 추가된 상태 객체.
            """
            print(f"[GuideGraphBuilder] PDF 로딩 시작: {st.url}", flush=True)
            st.chunks = await self.loader.load(st.url, with_figures=True)
            print(f"[GuideGraphBuilder] PDF 로딩 완료: {len(st.chunks)}개 청크", flush=True)
            return st

        g.add_node("load", load_pdf)

        # 2. Generate sections -------------------------------------------
        @safe_retry
        async def generate_sections(st: GuideState):
            """자습서 섹션들을 생성한다.

            Args:
                st: 현재 요청 상태.

            Returns:
                생성된 섹션들이 추가된 상태 객체.
            """
            if not st.chunks:
                raise ValueError("PDF 청크가 없습니다. PDF 로딩에 실패했거나 내용이 비어있습니다.")
            
            if not isinstance(st.chunks[0], PageChunk):
                raise ValueError("잘못된 청크 형식입니다. PageChunk 객체가 필요합니다.")
            
            sections = []
            
            # SemanticGrouper를 사용한 청크 그룹화
            print(f"[GuideGraphBuilder] 청크 그룹화 시작: {len(st.chunks)}개 청크", flush=True)
            groups = self.grouper.group_chunks(st.chunks)
            st.log.append(f"청크 그룹화 완료: {len(groups)}개 그룹")
            print(f"[GuideGraphBuilder] 청크 그룹화 완료: {len(groups)}개 그룹", flush=True)
            
            # 각 그룹별로 섹션 생성
            for i, grp in enumerate(groups, 1):
                st.log.append(f"그룹 {i} 처리 중: {len(grp)}개 청크")
                print(f"[GuideGraphBuilder] 그룹 {i} 처리 중: {len(grp)}개 청크", flush=True)
                
                # 청크 텍스트를 그대로 LLM에 전달 ([IMG_id:caption] 형태)
                chunks_text = "\n".join(c.text for c in grp)[:6_000]
                prompt = PROMPT_TUTORIAL.render(chunks=chunks_text)
                
                section = await self.llm.execute(prompt)
                
                # 이미지 ID는 그대로 유지 (finish에서 처리)
                sections.append(section)
                
                st.log.append(f"그룹 {i} 섹션 생성 완료")
                print(f"[GuideGraphBuilder] 그룹 {i} 섹션 생성 완료", flush=True)

            st.sections = sections
            print(f"[GuideGraphBuilder] 모든 섹션 생성 완료: {len(sections)}개", flush=True)
            return st

        g.add_node("generate", generate_sections)

        # 3. Combine sections --------------------------------------------
        @safe_retry
        async def combine_sections(st: GuideState):
            """섹션들을 하나의 일관된 Markdown 문서로 통합하고 필요시 번역한다.

            Args:
                st: 현재 요청 상태.

            Returns:
                통합된 튜토리얼이 추가된 상태 객체.
            """
            if not st.sections:
                raise ValueError("sections is empty — cannot combine")
            
            print(f"[GuideGraphBuilder] 섹션별 번역 시작: {len(st.sections)}개 섹션", flush=True)
            
            # 각 섹션을 개별적으로 번역
            translated_sections = []
            total_original_length = 0
            total_original_images = 0
            
            for i, section in enumerate(st.sections):
                if not section.strip():
                    continue
                
                # 번역 전 섹션 정보
                section_length = len(section)
                section_images = section.count('[IMG_')
                total_original_length += section_length
                total_original_images += section_images
                
                print(f"[GuideGraphBuilder] 섹션 {i+1} 번역 중: {section_length}자, {section_images}개 이미지", flush=True)
                
                # 각 섹션별로 번역
                section_prompt = PROMPT_TUTORIAL_TRANSLATE.render(lang=st.lang, text=section)
                translated_section = await self.llm.execute(section_prompt)
                translated_sections.append(translated_section)
                
                print(f"[GuideGraphBuilder] 섹션 {i+1} 번역 완료", flush=True)
            
            # 번역된 섹션들을 문서 구조로 결합
            translated_doc = self._create_document_structure(translated_sections)
            
            # 번역 후 내용 길이 및 구조 확인
            translated_length = len(translated_doc)
            translated_sections_count = translated_doc.count('#')
            translated_images = translated_doc.count('[IMG_')
            print(f"[GuideGraphBuilder] 번역 후: {translated_length}자, {len(translated_sections)}개 섹션, {translated_images}개 이미지", flush=True)
            
            # 내용 보존 확인
            if translated_length < total_original_length * 0.8:  # 20% 이상 줄어들면 경고
                print(f"[GuideGraphBuilder] ⚠️  경고: 번역 후 내용이 {total_original_length}자에서 {translated_length}자로 줄어듦", flush=True)
                st.log.append(f"번역 경고: 내용 길이 감소 ({total_original_length}→{translated_length}자)")
            
            if len(translated_sections) < len(st.sections):
                print(f"[GuideGraphBuilder] ⚠️  경고: 섹션 수 감소 ({len(st.sections)}→{len(translated_sections)})", flush=True)
                st.log.append(f"번역 경고: 섹션 수 감소 ({len(st.sections)}→{len(translated_sections)})")
            
            if translated_images < total_original_images:
                print(f"[GuideGraphBuilder] ⚠️  경고: 이미지 참조 감소 ({total_original_images}→{translated_images})", flush=True)
                st.log.append(f"번역 경고: 이미지 참조 감소 ({total_original_images}→{translated_images})")
            
            st.tutorial = translated_doc
            print(f"[GuideGraphBuilder] 번역 완료", flush=True)
            
            return st

        g.add_node("combine", combine_sections)

        # 4. Translate tutorial ------------------------------------------
        # translate_tutorial 노드는 combine_sections에서 처리하므로 제거

        # Routing -------------------------------------------------------
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

        def post_combine(st: GuideState) -> str:
            return "finish" if st.error else "finish"

        g.add_conditional_edges("combine", post_combine, {
            "finish": "finish",
        })

        # Finish node 추가
        async def finish_node(st: GuideState):
            """튜토리얼 생성 프로세스가 종료되면 실행 로그를 기록하고 이미지 ID를 URI로 교체한다."""
            if st.error:
                st.log.append(f"에러로 종료: {st.error}")
                print(f"[GuideGraphBuilder] 에러로 종료: {st.error}", flush=True)
            else:
                # 이미지 ID를 URI로 교체
                if st.tutorial and st.chunks:
                    print(f"[GuideGraphBuilder] 이미지 ID를 URI로 교체 시작", flush=True)
                    
                    # 디버깅: chunks 정보 출력
                    print(f"[GuideGraphBuilder] chunks 개수: {len(st.chunks)}", flush=True)
                    for i, chunk in enumerate(st.chunks):
                        print(f"[GuideGraphBuilder] chunk {i}: figs 개수 = {len(chunk.figs)}", flush=True)
                        for img_id, uri in chunk.figs:
                            print(f"[GuideGraphBuilder]   - {img_id} -> {uri[:50]}...", flush=True)
                    
                    # _create_image_mapping 메서드를 사용하여 이미지 매핑 생성
                    all_image_mapping = self._create_image_mapping(st.chunks)
                    print(f"[GuideGraphBuilder] 생성된 이미지 매핑: {len(all_image_mapping)}개", flush=True)
                    for img_id, uri in all_image_mapping.items():
                        print(f"[GuideGraphBuilder]   매핑: {img_id} -> {uri[:50]}...", flush=True)
                    
                    # 디버깅: tutorial에서 이미지 ID 패턴 확인
                    import re
                    img_patterns = re.findall(r'\[(IMG_\d+_\d+)\]', st.tutorial)
                    print(f"[GuideGraphBuilder] tutorial에서 찾은 이미지 ID 패턴: {img_patterns}", flush=True)
                    
                    # 튜토리얼에서 이미지 ID를 URI로 교체
                    st.tutorial = self._replace_image_ids_with_uris(st.tutorial, all_image_mapping)
                    
                    replaced_count = len(all_image_mapping)
                    st.log.append(f"이미지 ID 교체 완료: {replaced_count}개 이미지")
                    print(f"[GuideGraphBuilder] 이미지 ID 교체 완료: {replaced_count}개 이미지", flush=True)
                
                st.log.append("튜토리얼 생성 완료")
                print(f"[GuideGraphBuilder] 튜토리얼 생성 완료", flush=True)
            return st

        g.add_node("finish", finish_node)
        # translate 노드는 combine에서 처리하므로 제거

        g.set_finish_point("finish")
        return g.compile()

    # ──────────────── Helper methods ────────────────
    def _create_image_mapping(self, chunks: List[PageChunk]) -> dict:
        """청크에서 이미지 ID를 URI로 매핑하는 딕셔너리 생성."""
        image_mapping = {}
        
        for chunk in chunks:
            for img_id, uri in chunk.figs:
                image_mapping[img_id] = uri
        
        return image_mapping

    def _replace_image_ids_with_uris(self, section: str, image_mapping: dict) -> str:
        """섹션에서 이미지 ID [IMG_X]를 실제 URI로 교체."""
        import re
        
        print(f"[GuideGraphBuilder] _replace_image_ids_with_uris 시작: 매핑 {len(image_mapping)}개", flush=True)
        
        # [IMG_X] 패턴을 찾아서 실제 이미지 태그로 교체
        def replace_image_id(match):
            img_id = match.group(1)
            print(f"[GuideGraphBuilder] 매칭된 이미지 ID: {img_id}", flush=True)
            if img_id in image_mapping:
                uri = image_mapping[img_id]
                print(f"[GuideGraphBuilder] 매핑 성공: {img_id} -> {uri[:50]}...", flush=True)
                return f"![figure]({uri})"
            else:
                print(f"[GuideGraphBuilder] 매핑 실패: {img_id} (매핑에 없음)", flush=True)
                return match.group(0)  # 매핑이 없으면 원본 유지
        
        # [IMG_X_Y] 패턴을 찾아서 교체 (페이지_번호 형식)
        section_with_images = re.sub(r'\[(IMG_\d+_\d+)\]', replace_image_id, section)
        
        print(f"[GuideGraphBuilder] _replace_image_ids_with_uris 완료", flush=True)
        return section_with_images

    def _create_document_structure(self, sections: List[str]) -> str:
        """섹션들을 하나의 일관된 Markdown 문서로 통합한다."""
        if not sections:
            return ""
            
        # 문서 시작
        combined = "# 자습서 가이드\n\n"
        
        # 목차 생성
        combined += "## 목차\n\n"
        for i, section in enumerate(sections, 1):
            # 섹션에서 제목 추출
            title = self._extract_section_title(section)
            if title:
                # URL 친화적인 앵커 생성
                anchor = self._create_anchor(title)
                combined += f"{i}. [{title}](#{anchor})\n"
                
                # 하위 섹션들도 목차에 추가
                subsections = self._extract_subsections(section)
                for j, subsection in enumerate(subsections, 1):
                    sub_anchor = self._create_anchor(subsection)
                    combined += f"   {i}.{j}. [{subsection}](#{sub_anchor})\n"
            else:
                combined += f"{i}. [섹션 {i}](#section-{i})\n"
        combined += "\n---\n\n"
        
        # 섹션들을 순서대로 추가
        for i, section in enumerate(sections, 1):
            # 섹션 번호와 제목 추가
            title = self._extract_section_title(section)
            if title:
                section_with_number = f"## {i}. {title}\n\n"
                # 원본 섹션에서 첫 번째 헤더 제거하고 나머지 내용만 추가
                content = self._remove_first_header(section)
                combined += section_with_number + content
            else:
                # 제목이 없으면 섹션 번호만 추가
                combined += f"## {i}. 섹션\n\n{section.strip()}\n\n"
            
            # 섹션 간 구분자 (마지막 섹션 제외)
            if i < len(sections):
                combined += "---\n\n"
        
        return combined

    def _extract_section_title(self, section: str) -> str:
        """섹션에서 제목을 추출한다."""
        lines = section.strip().split('\n')
        for line in lines:
            line = line.strip()
            # 마크다운 헤더에서 제목 추출
            if line.startswith('#') and len(line) > 1:
                return line.lstrip('#').strip()
        return ""

    def _remove_first_header(self, section: str) -> str:
        """섹션에서 첫 번째 헤더를 제거한다."""
        lines = section.strip().split('\n')
        result_lines = []
        header_removed = False
        
        for line in lines:
            if not header_removed and line.strip().startswith('#') and len(line.strip()) > 1:
                header_removed = True
                continue
            result_lines.append(line)
        
        return '\n'.join(result_lines).strip()

    def _create_anchor(self, title: str) -> str:
        """제목에서 URL 친화적인 앵커를 생성한다."""
        import re
        # 소문자로 변환하고 특수문자 제거
        anchor = re.sub(r'[^\w\s-]', '', title.lower())
        # 공백을 하이픈으로 변환
        anchor = re.sub(r'[-\s]+', '-', anchor)
        return anchor

    def _extract_subsections(self, section: str) -> List[str]:
        """섹션에서 하위 섹션 제목들을 추출한다."""
        subsections = []
        lines = section.strip().split('\n')
        for line in lines:
            line = line.strip()
            # H2 헤더 (##)를 하위 섹션으로 인식
            if line.startswith('##') and len(line) > 2:
                title = line.lstrip('#').strip()
                subsections.append(title)
        return subsections

