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
            
            # ✅ Docling으로 PDF 변환 (직접 객체 사용)
            doc = _converter.convert(source=url).document
            
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            
            # 페이지 정보 확인
            num_pages = len(doc.pages) if hasattr(doc, 'pages') else 0
            print(f"[PDFReceiver] PDF 변환 완료: {num_pages}개 페이지 ({processing_time:.2f}초)", flush=True)
            
            # GPU 메모리 사용량 모니터링
            if torch.cuda.is_available():
                end_memory = torch.cuda.memory_allocated(0)
                memory_used = (end_memory - start_memory) / 1024**3
                print(f"[PDFReceiver] GPU 메모리 사용량 변화: {memory_used:.2f}GB", flush=True)
            
        except Exception as e:
            raise ValueError(f"Docling PDF 변환 실패: {e}")

        # ✅ 새로운 방식: doc 객체에서 직접 요소 추출 (bbox 기반 정렬)
        elements: List[PageElement] = []
        
        # 페이지별로 요소를 그룹화 (bbox 포함)
        from collections import defaultdict
        page_items = defaultdict(lambda: {"items": []})  # items: [(y_position, type, data), ...]
        
        # 1. 텍스트 추출 (bbox 정보 포함)
        if hasattr(doc, 'texts'):
            print(f"[PDFReceiver] 텍스트 요소 추출 중: {len(doc.texts)}개", flush=True)
            for text_item in doc.texts:
                if text_item.prov:
                    page_no = text_item.prov[0].page_no
                    # 텍스트가 있는 경우만 추가
                    if hasattr(text_item, 'text') and text_item.text.strip():
                        # bbox가 있으면 y좌표 추출, 없으면 큰 값(끝으로 배치)
                        y_pos = text_item.prov[0].bbox.t if hasattr(text_item.prov[0], 'bbox') and text_item.prov[0].bbox else 999999
                        page_items[page_no]["items"].append((y_pos, "text", text_item.text.strip()))
        
        # 2. 이미지 추출 (bbox 정보 포함)
        if hasattr(doc, 'pictures'):
            print(f"[PDFReceiver] 이미지 요소 추출 중: {len(doc.pictures)}개", flush=True)
            for pic_item in doc.pictures:
                if pic_item.prov:
                    page_no = pic_item.prov[0].page_no
                    # 이미지가 있는 경우만 추가
                    if hasattr(pic_item, 'image') and pic_item.image:
                        # base64 data-URI로 변환
                        if isinstance(pic_item.image.pil_image, bytes):
                            img_bytes = pic_item.image.pil_image
                        else:
                            # PIL Image → bytes
                            import io
                            from PIL import Image
                            buffer = io.BytesIO()
                            pic_item.image.pil_image.save(buffer, format='PNG')
                            img_bytes = buffer.getvalue()
                        
                        # 캡션 추출 (있으면)
                        caption = ""
                        if hasattr(pic_item, 'captions') and pic_item.captions:
                            caption = pic_item.captions[0].text if hasattr(pic_item.captions[0], 'text') else ""
                        
                        # bbox가 있으면 y좌표 추출, 없으면 큰 값(끝으로 배치)
                        y_pos = pic_item.prov[0].bbox.t if hasattr(pic_item.prov[0], 'bbox') and pic_item.prov[0].bbox else 999999
                        page_items[page_no]["items"].append((y_pos, "image", (img_bytes, caption)))
        
        # 3. 페이지별로 PageElement 생성 (bbox 순서대로)
        print(f"[PDFReceiver] 페이지별 요소 생성 시작: {len(page_items)}개 페이지", flush=True)
        
        for page_no in sorted(page_items.keys()):
            items = page_items[page_no]["items"]
            
            # ✅ bbox의 y좌표(top) 기준으로 정렬 - 원본 PDF의 배치 유지
            sorted_items = sorted(items, key=lambda x: x[0])
            
            # 이미지 카운터 초기화
            image_counter = 1
            
            # 통계
            num_texts = sum(1 for _, item_type, _ in sorted_items if item_type == "text")
            num_images = sum(1 for _, item_type, _ in sorted_items if item_type == "image")
            print(f"[PDFReceiver] 페이지 {page_no} 처리: 텍스트 {num_texts}개, 이미지 {num_images}개 (bbox 정렬됨)", flush=True)
            
            # bbox 순서대로 텍스트와 이미지를 처리
            text_buffer = []
            
            for y_pos, item_type, data in sorted_items:
                if item_type == "text":
                    # 텍스트 추가
                    text_buffer.append(data)
                    
                elif item_type == "image":
                    # 이미지를 만나면 텍스트 버퍼를 먼저 flush
                    if text_buffer:
                        text_content = "\n\n".join(text_buffer)
                        for para in re.split(r"\n{2,}", text_content):
                            if para.strip():
                                elements.append(PageElement("text", page_no, para.strip()))
                        text_buffer.clear()
                    
                    # 이미지 데이터 추가 (플레이스홀더는 semantic_chunker가 자동 추가)
                    img_bytes, caption = data
                    img_id = f"IMG_{page_no}_{image_counter}"
                    
                    print(f"[PDFReceiver] 이미지 추가: {img_id} (y={y_pos:.1f})", flush=True)
                    
                    # ✅ 이미지 데이터만 추가 (플레이스홀더는 별도 추가 안 함)
                    # semantic_chunker가 figure를 보고 [IMG_X_Y] 플레이스홀더를 텍스트에 자동 삽입
                    elements.append(PageElement("figure", page_no, img_bytes, caption=caption or "Figure", id=img_id))
                    
                    image_counter += 1
            
            # 마지막 남은 텍스트 처리
            if text_buffer:
                text_content = "\n\n".join(text_buffer)
                for para in re.split(r"\n{2,}", text_content):
                    if para.strip():
                        elements.append(PageElement("text", page_no, para.strip()))

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
