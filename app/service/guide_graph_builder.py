# app/service/guide_graph_builder.py
"""GuideGraphBuilder – 멀티모달 PDF → 튜토리얼 Markdown
도메인 기반 LangGraph 파이프라인 빌더.
모든 노드는 최대 3회 재시도하며, 실패 시 에러 메시지를 기록하고 종료한다.
"""

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
from app.prompts import PROMPT_TUTORIAL, PROMPT_TUTORIAL_TRANSLATE

# ──────────────── 환경변수 설정 ────────────────
DEBUG = os.getenv("DEBUG_TUTORIAL", "false").lower() == "true"

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

    # ──────────────── Helper methods ────────────────
    def _safe_truncate(self, text: str, max_len: int = 6000) -> str:
        """이미지 참조 경계를 존중하는 안전한 텍스트 트러케이션.
        
        Args:
            text: 트러케이션할 텍스트
            max_len: 최대 길이
            
        Returns:
            트러케이션된 텍스트 (이미지 참조가 잘리지 않음)
        """
        if len(text) <= max_len:
            return text
        
        # max_len 근처에서 이미지 참조가 아닌 곳에서 자르기
        truncated = text[:max_len]
        
        # 마지막 완전한 이미지 참조 찾기 ([IMG_X_Y:caption] 형태)
        last_img_end = truncated.rfind(']')
        if last_img_end != -1:
            # 이미지 참조 경계에서 자르기
            return truncated[:last_img_end + 1]
        
        # 이미지 참조가 없으면 그냥 자르기
        return truncated

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
            """자습서 섹션들을 생성한다 (병렬 처리).

            Args:
                st: 현재 요청 상태.

            Returns:
                생성된 섹션들이 추가된 상태 객체.
            """
            if not st.chunks:
                raise ValueError("PDF 청크가 없습니다. PDF 로딩에 실패했거나 내용이 비어있습니다.")
            
            if not isinstance(st.chunks[0], PageChunk):
                raise ValueError("잘못된 청크 형식입니다. PageChunk 객체가 필요합니다.")
            
            # SemanticGrouper를 사용한 청크 그룹화
            print(f"[GuideGraphBuilder] 청크 그룹화 시작: {len(st.chunks)}개 청크", flush=True)
            groups = self.grouper.group_chunks(st.chunks)
            st.log.append(f"청크 그룹화 완료: {len(groups)}개 그룹")
            print(f"[GuideGraphBuilder] 청크 그룹화 완료: {len(groups)}개 그룹", flush=True)
            
            # 병렬 섹션 생성
            async def _generate_one_section(grp, idx):
                """단일 섹션을 생성한다."""
                try:
                    if DEBUG:
                        print(f"[GuideGraphBuilder] 그룹 {idx} 처리 중: {len(grp)}개 청크", flush=True)
                    
                    # 안전한 트러케이션으로 이미지 참조 보존
                    chunks_text = self._safe_truncate("\n".join(c.text for c in grp), 6_000)
                    prompt = PROMPT_TUTORIAL.render(chunks=chunks_text)
                    
                    section = await self.llm.execute(prompt)
                    
                    if DEBUG:
                        print(f"[GuideGraphBuilder] 그룹 {idx} 섹션 생성 완료", flush=True)
                    
                    return idx, section
                except Exception as e:
                    print(f"[GuideGraphBuilder] ⚠️ 그룹 {idx} 섹션 생성 실패: {e}", flush=True)
                    st.log.append(f"섹션 {idx} 생성 실패")
                    # 실패 시 기본 섹션 반환
                    return idx, f"## 섹션 {idx}\n\n(생성 실패)"
            
            # 모든 그룹을 병렬로 처리
            print(f"[GuideGraphBuilder] {len(groups)}개 섹션 병렬 생성 시작", flush=True)
            tasks = [_generate_one_section(grp, i) for i, grp in enumerate(groups, 1)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 정렬 및 수집
            sections = []
            for result in sorted(results, key=lambda x: x[0] if isinstance(x, tuple) else 999):
                if isinstance(result, Exception):
                    print(f"[GuideGraphBuilder] ⚠️ 섹션 생성 예외: {result}", flush=True)
                    sections.append("## 섹션\n\n(생성 실패)")
                elif isinstance(result, tuple):
                    idx, section = result
                    sections.append(section)
                    st.log.append(f"그룹 {idx} 섹션 생성 완료")

            st.sections = sections
            print(f"[GuideGraphBuilder] 모든 섹션 생성 완료: {len(sections)}개", flush=True)
            return st

        g.add_node("generate", generate_sections)

        # 3. Combine sections --------------------------------------------
        @safe_retry
        async def combine_sections(st: GuideState):
            """섹션들을 하나의 일관된 Markdown 문서로 통합하고 필요시 번역한다 (병렬 처리).

            Args:
                st: 현재 요청 상태.

            Returns:
                통합된 튜토리얼이 추가된 상태 객체.
            """
            if not st.sections:
                raise ValueError("sections is empty — cannot combine")
            
            # 빈 섹션 필터링 (섹션 번호 정확성 보장)
            valid_sections = [s for s in st.sections if s.strip()]
            if not valid_sections:
                raise ValueError("모든 섹션이 비어있습니다")
            
            print(f"[GuideGraphBuilder] 섹션별 번역 시작: {len(valid_sections)}개 유효 섹션", flush=True)
            
            # 통계 수집
            total_original_length = sum(len(s) for s in valid_sections)
            total_original_images = sum(s.count('[IMG_') for s in valid_sections)
            
            # 병렬 번역
            async def _translate_one_section(section, idx):
                """단일 섹션을 번역한다."""
                try:
                    section_length = len(section)
                    section_images = section.count('[IMG_')
                    
                    if DEBUG:
                        print(f"[GuideGraphBuilder] 섹션 {idx+1} 번역 중: {section_length}자, {section_images}개 이미지", flush=True)
                    
                    section_prompt = PROMPT_TUTORIAL_TRANSLATE.render(lang=st.lang, text=section)
                    translated_section = await self.llm.execute(section_prompt)
                    
                    if DEBUG:
                        print(f"[GuideGraphBuilder] 섹션 {idx+1} 번역 완료", flush=True)
                    
                    return idx, translated_section
                except Exception as e:
                    # 번역 실패 시 원본 섹션 사용
                    print(f"[GuideGraphBuilder] ⚠️ 섹션 {idx+1} 번역 실패, 원본 사용: {e}", flush=True)
                    st.log.append(f"섹션 {idx+1} 번역 실패 (원본 사용)")
                    return idx, section  # 원본 그대로
            
            # 모든 섹션을 병렬로 번역
            print(f"[GuideGraphBuilder] {len(valid_sections)}개 섹션 병렬 번역 시작", flush=True)
            tasks = [_translate_one_section(s, i) for i, s in enumerate(valid_sections)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 정렬 및 수집
            translated_sections = []
            for result in sorted(results, key=lambda x: x[0] if isinstance(x, tuple) else 999):
                if isinstance(result, Exception):
                    print(f"[GuideGraphBuilder] ⚠️ 번역 예외: {result}", flush=True)
                    # 예외 발생 시 기본 텍스트
                    translated_sections.append("## 섹션\n\n(번역 실패)")
                elif isinstance(result, tuple):
                    idx, section = result
                    translated_sections.append(section)
            
            # 번역된 섹션들을 문서 구조로 결합
            translated_doc = self._create_document_structure(translated_sections)
            
            # 번역 후 내용 길이 및 구조 확인
            translated_length = len(translated_doc)
            translated_images = translated_doc.count('[IMG_')
            print(f"[GuideGraphBuilder] 번역 후: {translated_length}자, {len(translated_sections)}개 섹션, {translated_images}개 이미지", flush=True)
            
            # 내용 보존 확인
            if translated_length < total_original_length * 0.8:  # 20% 이상 줄어들면 경고
                print(f"[GuideGraphBuilder] ⚠️  경고: 번역 후 내용이 {total_original_length}자에서 {translated_length}자로 줄어듦", flush=True)
                st.log.append(f"번역 경고: 내용 길이 감소 ({total_original_length}→{translated_length}자)")
            
            if len(translated_sections) < len(valid_sections):
                print(f"[GuideGraphBuilder] ⚠️  경고: 섹션 수 감소 ({len(valid_sections)}→{len(translated_sections)})", flush=True)
                st.log.append(f"번역 경고: 섹션 수 감소 ({len(valid_sections)}→{len(translated_sections)})")
            
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

        # combine → finish 직접 연결 (조건 불필요)
        g.add_edge("combine", "finish")

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
                    
                    # 이미지 매핑 생성
                    all_image_mapping = self._create_image_mapping(st.chunks)
                    
                    if DEBUG:
                        # 디버깅: chunks 정보 출력
                        print(f"[GuideGraphBuilder] chunks 개수: {len(st.chunks)}", flush=True)
                        for i, chunk in enumerate(st.chunks):
                            print(f"[GuideGraphBuilder] chunk {i}: figs 개수 = {len(chunk.figs)}", flush=True)
                            for img_id, uri in chunk.figs:
                                print(f"[GuideGraphBuilder]   - {img_id} -> {uri[:50]}...", flush=True)
                        
                        print(f"[GuideGraphBuilder] 생성된 이미지 매핑: {len(all_image_mapping)}개", flush=True)
                        for img_id, uri in all_image_mapping.items():
                            print(f"[GuideGraphBuilder]   매핑: {img_id} -> {uri[:50]}...", flush=True)
                        
                        # tutorial에서 이미지 ID 패턴 확인
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
        """섹션에서 이미지 ID [IMG_X] 또는 [IMG_X:caption]을 실제 URI로 교체."""
        if DEBUG:
            print(f"[GuideGraphBuilder] _replace_image_ids_with_uris 시작: 매핑 {len(image_mapping)}개", flush=True)
        
        # [IMG_X] 또는 [IMG_X:caption] 패턴을 찾아서 실제 이미지 태그로 교체
        def replace_image_id(match):
            img_id = match.group(1)
            caption_text = match.group(2) if match.lastindex >= 2 else None
            
            if DEBUG:
                print(f"[GuideGraphBuilder] 매칭된 이미지 ID: {img_id} (캡션: {caption_text[:30] if caption_text else 'None'})", flush=True)
            
            if img_id in image_mapping:
                uri = image_mapping[img_id]
                if DEBUG:
                    print(f"[GuideGraphBuilder] 매핑 성공: {img_id} -> {uri[:50]}...", flush=True)
                
                # 캡션이 있으면 alt 텍스트로 사용
                if caption_text:
                    # 캡션에서 특수문자 제거 (alt 텍스트용)
                    alt_text = caption_text.strip()[:100]  # 100자 제한
                    return f"![{alt_text}]({uri})"
                else:
                    return f"![figure]({uri})"
            else:
                # 매핑 실패 시 경고
                print(f"[GuideGraphBuilder] ⚠️ 매핑 실패: {img_id} (매핑에 없음)", flush=True)
                return f"![이미지 로드 실패]() <!-- {img_id} 매핑 없음 -->"
        
        # [IMG_X_Y] 또는 [IMG_X_Y:caption] 패턴을 찾아서 교체
        # 정규식: [IMG_숫자_숫자] 또는 [IMG_숫자_숫자:캡션내용]
        section_with_images = re.sub(
            r'\[(IMG_\d+_\d+)(?::([^\]]+))?\]', 
            replace_image_id, 
            section
        )
        
        if DEBUG:
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

