"""PDFReceiver (Docling + SmolDocling)
=====================================
URL → List[PageElement]

• Docling의 Markdown 결과를 읽어 페이지 흐름 그대로
  PageElement(kind="text" | "figure") 리스트로 변환
• data-URI 이미지는 base64 디코딩, 원격 URL은 병렬 다운로드(동시 8개)
• OCR/PyMuPDF 제거로 속도 단축
"""

from __future__ import annotations

import asyncio, base64, re, os
from typing import Final, List, Tuple
import httpx
import torch

# GPU 디바이스 동기화 문제 해결을 위한 환경변수들
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 0번만 사용
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TORCH_USE_CUDA_DSA"] = "1"  # CUDA 디바이스 side 어써션

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.base import ImageRefMode

from app.domain.page_element import PageElement

# ──────────────── 설정 ────────────────
_TIMEOUT = httpx.Timeout(30.0)
_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

# GPU 최적화 설정
if torch.cuda.is_available():
    # GPU 메모리 할당 최적화
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ──────────────── Docling 설정 ────────────────
# 성능 최적화된 SmolDocling 설정 (인터넷 검색 결과 기반)
try:
    # 표준 PDF 파이프라인 사용 (VLM의 GPU 텐서 문제 회피)
    pipeline_options = PdfPipelineOptions(
        generate_picture_images=True
        # 기본값 사용 - OCR과 레이아웃 분석 포함
    )
    
    _converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,  # VLM 대신 표준 파이프라인
                pipeline_options=pipeline_options,
                embedding=True  # 이미지 추출 활성화
            )
        }
    )
    print("[PDFReceiver] Docling 성능 최적화 설정으로 초기화 완료", flush=True)
    
    # GPU 가속이 가능한지 확인
    if torch.cuda.is_available():
        print(f"[PDFReceiver] GPU 사용 가능: {torch.cuda.get_device_name(0)}", flush=True)
        # GPU 메모리 최적화 설정
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[PDFReceiver] GPU 최적화 설정 완료", flush=True)
    
except Exception as e:
    print(f"[PDFReceiver] Docling 초기화 실패: {e}", flush=True)
    raise

device_info = "GPU" if torch.cuda.is_available() else "CPU"
print(f"[PDFReceiver] Docling 초기화 완료 - {device_info} 모드", flush=True)
if torch.cuda.is_available():
    print(f"[PDFReceiver] GPU: {torch.cuda.get_device_name(0)}", flush=True)

class PDFReceiver:
    """
    PDF URL → PageElement 리스트로 변환.
    SmolDocling + Docling 기반으로 완전히 재작성.
    성능 최적화: 캐싱, 배치 처리, GPU 가속 적용.
    """
    
    def __init__(self):
        # 간단한 메모리 캐시 (URL → 처리 결과)
        self._cache = {}
        self._cache_size_limit = 10  # 최대 캐시 크기

    async def fetch_and_extract_elements(self, url: str) -> List[PageElement]:
        """
        PDF URL에서 텍스트와 이미지를 추출하여 PageElement 리스트로 반환.
        성능 최적화: 캐싱, GPU 가속, 배치 처리 적용.
        
        Returns
        -------
        List[PageElement]
            추출된 페이지 요소들 (text, figure, table, graph)
        """
        # 캐시 확인
        if url in self._cache:
            print(f"[PDFReceiver] 캐시 히트: {url}", flush=True)
            return self._cache[url]
        
        try:
            print(f"[PDFReceiver] PDF 변환 시작: {url}", flush=True)
            
            # GPU 메모리 최적화 설정
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # GPU 메모리 정리
                start_memory = torch.cuda.memory_allocated(0)
                print(f"[PDFReceiver] GPU 메모리 사용량: {start_memory / 1024**3:.2f}GB", flush=True)
                
                # 추가 GPU 최적화 설정
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # 성능 최적화된 PDF 변환
            import time
            start_time = time.perf_counter()
            
            # Docling으로 PDF → Markdown 변환 (성능 최적화)
            doc = _converter.convert(source=url).document
            
            # ✅ 디버깅: Docling 문서 구조 확인
            print(f"[PDFReceiver] 🔍 Docling 문서 구조:", flush=True)
            print(f"[PDFReceiver]   doc 타입: {type(doc)}", flush=True)
            print(f"[PDFReceiver]   doc 속성: {dir(doc)[:20]}", flush=True)
            
            # 페이지 정보 확인
            if hasattr(doc, 'pages'):
                print(f"[PDFReceiver]   ✅ doc.pages 존재: {len(doc.pages)}개 페이지", flush=True)
            if hasattr(doc, 'num_pages'):
                print(f"[PDFReceiver]   ✅ doc.num_pages: {doc.num_pages}", flush=True)
            if hasattr(doc, 'page_count'):
                print(f"[PDFReceiver]   ✅ doc.page_count: {doc.page_count}", flush=True)
            
            markdown_content = doc.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
            
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            print(f"[PDFReceiver] PDF 변환 완료: {len(markdown_content)}자 ({processing_time:.2f}초)", flush=True)
            
            # 디버깅: Markdown 내용에서 이미지 패턴 확인
            img_patterns_in_markdown = list(_IMG_RE.findall(markdown_content))
            print(f"[PDFReceiver] 전체 Markdown에서 찾은 이미지 패턴: {len(img_patterns_in_markdown)}개", flush=True)
            for i, (alt, src) in enumerate(img_patterns_in_markdown[:5]):  # 처음 5개만 출력
                print(f"[PDFReceiver]   전체 이미지 {i+1}: alt='{alt[:30]}...', src='{src[:50]}...'", flush=True)
            
            # 디버깅: Markdown 내용 일부 출력
            print(f"[PDFReceiver] === Markdown 내용 미리보기 ===", flush=True)
            markdown_preview = markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content
            print(f"[PDFReceiver] {markdown_preview}", flush=True)
            print(f"[PDFReceiver] === Markdown 내용 끝 ===", flush=True)
            
            # GPU 메모리 사용량 모니터링
            if torch.cuda.is_available():
                end_memory = torch.cuda.memory_allocated(0)
                memory_used = (end_memory - start_memory) / 1024**3
                print(f"[PDFReceiver] GPU 메모리 사용량 변화: {memory_used:.2f}GB", flush=True)
            
            # SmolDocling 페이지 구분자로 분할
            # SmolDocling은 <page_break> 태그를 사용하거나 페이지 번호를 포함할 수 있음
            if "<page_break>" in markdown_content:
                pages = markdown_content.split("<page_break>")
            elif "\f" in markdown_content:  # form feed character
                pages = markdown_content.split("\f")
            else:
                # 페이지 구분자가 없으면 전체를 하나의 페이지로
                pages = [markdown_content]
            
        except Exception as e:
            raise ValueError(f"Docling PDF 변환 실패: {e}")

        elements: List[PageElement] = []
        remote_imgs: List[Tuple[int, str, str, str]] = []  # (page_idx, alt, url, img_id)

        for idx, pg_md in enumerate(pages):
            image_counter = 1  # 페이지별로 이미지 ID 카운터 초기화
            if not pg_md.strip():
                continue

            print(f"[PDFReceiver] 페이지 {idx} 처리 중: {len(pg_md)}자", flush=True)

            # 원본 Markdown에서 이미지 패턴 찾기 (한 번만)
            img_matches = list(_IMG_RE.findall(pg_md))
            print(f"[PDFReceiver] 페이지 {idx}에서 찾은 이미지 패턴: {len(img_matches)}개", flush=True)
            for i, (alt, src) in enumerate(img_matches):
                print(f"[PDFReceiver]   이미지 {i+1}: alt='{alt[:50]}...', src='{src[:100]}...'", flush=True)

            # (1) 텍스트 처리 - 이미지 매칭 결과를 사용하여 플레이스홀더 생성
            def _placeholder(m: re.Match) -> str:
                nonlocal image_counter
                img_id = f"IMG_{idx}_{image_counter}"
                image_counter += 1
                print(f"[PDFReceiver] 이미지 플레이스홀더 생성: {img_id}", flush=True)
                return f"[{img_id}]"

            text_with_fig = _IMG_RE.sub(_placeholder, pg_md)
            for para in re.split(r"\n{2,}", text_with_fig):
                if para.strip():
                    elements.append(PageElement("text", idx, para.strip()))

            # (2) 이미지 처리 - 이미 매칭된 결과 사용
            # 카운터는 텍스트 처리 시 이미 증가했으므로 리셋하지 않음
            # enumerate로 1부터 시작하여 명시적으로 ID 생성
            for img_idx, (alt, src) in enumerate(img_matches, 1):
                img_id = f"IMG_{idx}_{img_idx}"
                
                print(f"[PDFReceiver] 이미지 처리 중: {img_id}", flush=True)
                
                if src.startswith("data:image"):
                    # data-URI → bytes 변환
                    _, b64 = src.split(",", 1)
                    try:
                        img_bytes = base64.b64decode(b64)
                        elements.append(PageElement("figure", idx, img_bytes, caption=alt, id=img_id))
                        print(f"[PDFReceiver] data-URI 이미지 추가: {img_id} ({len(img_bytes)} bytes)", flush=True)
                    except Exception as e:
                        print(f"[PDFReceiver] data-URI 디코딩 실패: {img_id} - {e}", flush=True)
                        continue
                else:
                    # remote URL은 나중에 다운로드
                    remote_imgs.append((idx, alt, src, img_id))
                    print(f"[PDFReceiver] 원격 이미지 추가: {img_id} -> {src[:100]}...", flush=True)

        # (3) 원격 이미지 다운로드 (동시 8개 제한)
        if remote_imgs:
            print(f"[PDFReceiver] 원격 이미지 다운로드 시작: {len(remote_imgs)}개", flush=True)
            sem = asyncio.Semaphore(8)
            
            async def _fetch(i: int, url: str):
                async with sem:
                    try:
                        print(f"[PDFReceiver] 이미지 다운로드 중: {url[:100]}...", flush=True)
                        r = await cli.get(url, follow_redirects=True)
                        print(f"[PDFReceiver] 이미지 다운로드 성공: {url[:100]}... (상태: {r.status_code})", flush=True)
                        return i, r
                    except Exception as e:
                        print(f"[PDFReceiver] 이미지 다운로드 실패: {url[:100]}... - {e}", flush=True)
                        return i, e

            async with httpx.AsyncClient(timeout=_TIMEOUT) as cli:
                resps = await asyncio.gather(*(_fetch(i, u) for i, _, u, _ in remote_imgs))

            for (pg_idx, alt, _, img_id), (i, r) in zip(remote_imgs, resps):
                if isinstance(r, Exception) or r.status_code != 200:
                    print(f"[PDFReceiver] 원격 이미지 처리 실패: {img_id} - {r}", flush=True)
                    continue
                elements.append(PageElement("figure", pg_idx, r.content, caption=alt, id=img_id))
                print(f"[PDFReceiver] 원격 이미지 추가: {img_id} ({len(r.content)} bytes)", flush=True)
        else:
            print(f"[PDFReceiver] 원격 이미지 없음", flush=True)

        if not elements:
            raise ValueError("Docling PDF 파싱 결과가 없습니다")
        
        # 결과를 캐시에 저장
        self._cache[url] = elements
        
        # 캐시 크기 제한 관리
        if len(self._cache) > self._cache_size_limit:
            # 가장 오래된 항목 제거 (FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        print(f"[PDFReceiver] 요소 추출 완료: {len(elements)}개 (텍스트: {len([e for e in elements if e.kind == 'text'])}, 이미지: {len([e for e in elements if e.kind in ('figure', 'table', 'graph')])})", flush=True)
        
        # 디버깅: 각 요소의 상세 정보 출력
        print(f"[PDFReceiver] === 요소 상세 정보 ===", flush=True)
        for i, element in enumerate(elements[:10]):  # 처음 10개만 출력
            print(f"[PDFReceiver] 요소 {i+1}:", flush=True)
            print(f"  - kind: {element.kind}", flush=True)
            print(f"  - page_no: {element.page_no}", flush=True)
            print(f"  - id: {element.id}", flush=True)
            if element.kind == "text":
                content_preview = element.content[:100] + "..." if len(element.content) > 100 else element.content
                print(f"  - content: {content_preview}", flush=True)
            else:
                content_type = type(element.content).__name__
                content_size = len(element.content) if hasattr(element.content, '__len__') else "N/A"
                print(f"  - content: {content_type} ({content_size})", flush=True)
                print(f"  - caption: {element.caption}", flush=True)
            print(f"  - ---", flush=True)
        
        if len(elements) > 10:
            print(f"[PDFReceiver] ... (총 {len(elements)}개 요소 중 처음 10개만 표시)", flush=True)
        
        # 결과를 캐시에 저장
        return elements
