# -*- coding: utf-8 -*-
"""ColPali 비전 RAG 서비스 모듈

ColPali(멀티벡터 문서 임베딩) + Qwen2-VL(비전 LLM)을 사용한 문서 이미지 기반 RAG.
PDF의 각 페이지를 이미지로 변환하여 임베딩하고, 쿼리와 유사한 페이지를 검색한 후
VLM으로 답변을 생성합니다.
"""

import os
import re
import tempfile
import hashlib
from collections import defaultdict

import torch
import fitz  # PyMuPDF
import httpx
import chromadb
from PIL import Image
from chromadb.config import Settings
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.compression.token_pooling import HierarchicalTokenPooler


# ─────────────────────────────────────────────────────────────
# 환경 변수 / 기본 설정
# ─────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("COLPALI_MODEL", "vidore/colqwen2-v1.0")
DEVICE_MAP = os.getenv("COLPALI_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")

DEFAULT_POOL_FACTOR = int(os.getenv("COLPALI_POOL_FACTOR", "3"))
PER_TOKEN_CANDIDATES = int(os.getenv("COLPALI_CAND_PER_TOKEN", "300"))
TOPK_PAGES = int(os.getenv("COLPALI_TOPK", "5"))

VLM_NAME = os.getenv("VLM_NAME", "Qwen/Qwen2-VL-7B-Instruct")
TENSOR_DTYPE = torch.float32


# ─────────────────────────────────────────────────────────────
# In-memory Chroma 클라이언트 (ColPali 전용)
# ─────────────────────────────────────────────────────────────
class _TempVectorDB:
    """ColPali 전용 임시 Chroma 클라이언트 (텍스트 VectorDB와 분리)"""
    
    def __init__(self, persist_path=None):
        if persist_path:
            self.client = chromadb.PersistentClient(
                path=persist_path,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client(
                Settings(anonymized_telemetry=False, allow_reset=True)
            )


def get_vector_db():
    """ColPali용 Chroma 클라이언트 반환"""
    return _TempVectorDB(persist_path=None)


# ─────────────────────────────────────────────────────────────
# 유틸리티 함수
# ─────────────────────────────────────────────────────────────
def _ensure_local_pdf(path_or_url):
    """URL이면 다운로드, 로컬 경로면 그대로 반환
    
    Returns:
        tuple: (local_path, is_temp, original_url)
    """
    s = str(path_or_url)
    if re.match(r"^https?://", s, flags=re.I):
        with httpx.Client(timeout=30, follow_redirects=True) as c:
            r = c.get(s)
            r.raise_for_status()
        fp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        fp.write(r.content)
        fp.close()
        return fp.name, True, s
    return s, False, s


def _get_collection_name(url_or_path):
    """URL/경로에서 일관된 컬렉션명 생성 (URL은 해시, 로컬은 basename)"""
    if re.match(r"^https?://", str(url_or_path), flags=re.I):
        hash_obj = hashlib.md5(str(url_or_path).encode())
        return f"colpali_{hash_obj.hexdigest()[:16]}"
    else:
        return os.path.basename(url_or_path).replace('.pdf', '')


# ─────────────────────────────────────────────────────────────
# ColPali 서비스 클래스
# ─────────────────────────────────────────────────────────────
class Colpali:
    """ColPali 비전 RAG 서비스 (지연 로드 방식)"""
    
    _model_cache = None
    _processor_cache = None
    _pooler_cache = None
    
    def __init__(self):
        self.vdb = get_vector_db()
    
    @property
    def model(self):
        """ColPali 모델 (최초 사용 시 로드)"""
        if Colpali._model_cache is None:
            Colpali._model_cache = ColQwen2.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map=DEVICE_MAP,
                attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
            ).eval()
        return Colpali._model_cache
    
    @property
    def processor(self):
        """ColPali 프로세서 (최초 사용 시 로드)"""
        if Colpali._processor_cache is None:
            Colpali._processor_cache = ColQwen2Processor.from_pretrained(MODEL_NAME)
        return Colpali._processor_cache
    
    @property
    def pooler(self):
        """토큰 풀러 (최초 사용 시 생성)"""
        if Colpali._pooler_cache is None:
            Colpali._pooler_cache = HierarchicalTokenPooler()
        return Colpali._pooler_cache

    def pdf_to_images(self, url, dpi=200):
        """PDF를 페이지별 이미지로 변환 (PyMuPDF 사용)"""
        local_pdf, is_temp, orig_url = _ensure_local_pdf(url)
        
        # 영구 캐시 디렉토리 (URL 기반 고정 경로)
        pdf_basename = _get_collection_name(url)
        cache_dir = os.getenv("COLPALI_IMAGE_CACHE", "./colpali_images")
        out_dir = os.path.join(cache_dir, pdf_basename)
        os.makedirs(out_dir, exist_ok=True)
        
        try:
            doc = fitz.open(local_pdf)
            paths = []
            scale = dpi / 72.0
            mat = fitz.Matrix(scale, scale)
            
            for i, page in enumerate(doc):
                fp = os.path.join(out_dir, f"page_{i+1}.png")
                if not os.path.exists(fp):
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    pix.save(fp)
                paths.append(fp)
            
            doc.close()
            return paths
        finally:
            if is_temp:
                try:
                    os.remove(local_pdf)
                except:
                    pass

    def embed_images(self, images, pool_factor=DEFAULT_POOL_FACTOR, batch_size=4):
        """이미지를 배치 단위로 임베딩 (GPU 메모리 절약)"""
        all_pages = []
        
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i+batch_size]
            pil_imgs = [Image.open(p).convert("RGB") for p in batch_imgs]
            batch = self.processor.process_images(pil_imgs).to(self.model.device)
            
            with torch.no_grad():
                embs = self.model(**batch)
            
            pooled = self.pooler.pool_embeddings(
                embs,
                pool_factor=pool_factor,
                padding=True,
                padding_side=self.processor.tokenizer.padding_side,
            )
            
            pages = [e.to("cpu").to(dtype=TENSOR_DTYPE) for e in torch.unbind(pooled, dim=0)]
            all_pages.extend(pages)
            
            del batch, embs, pooled
            torch.cuda.empty_cache()
        
        return all_pages

    def embed_query(self, query):
        """쿼리를 멀티벡터 임베딩으로 변환"""
        with torch.no_grad():
            batch = self.processor.process_queries([query]).to(self.model.device)
            q = self.model(**batch)
        return q[0].to("cpu").to(dtype=TENSOR_DTYPE)

    def _ensure_indexed(self, pdf_path_or_url):
        """PDF가 색인되지 않았으면 색인 수행"""
        client = self.vdb.client
        local_pdf, is_temp, orig_url = _ensure_local_pdf(pdf_path_or_url)
        coll_name = _get_collection_name(pdf_path_or_url)
        
        try:
            col = client.get_or_create_collection(coll_name)
            if col.count() > 0:
                return

            images = self.pdf_to_images(pdf_path_or_url)
            page_embs = self.embed_images(images)

            ids, embs, metas, docs = [], [], [], []
            for page_idx, E in enumerate(page_embs):
                for tok_idx, vec in enumerate(E):
                    ids.append(f"{coll_name}:{page_idx}:{tok_idx}")
                    embs.append(vec.numpy().tolist())
                    metas.append({"doc_id": coll_name, "page": page_idx, "tok": tok_idx})
                    docs.append(images[page_idx])

            # 배치 단위로 Chroma에 추가
            BATCH = min(int(os.getenv("CHROMA_ADD_BATCH", "2000")), 5400)
            for i in range(0, len(ids), BATCH):
                col.add(
                    ids=ids[i:i+BATCH],
                    embeddings=embs[i:i+BATCH],
                    metadatas=metas[i:i+BATCH],
                    documents=docs[i:i+BATCH],
                )
        finally:
            if is_temp:
                try:
                    os.remove(local_pdf)
                except:
                    pass

    async def retrieve(self, url, query, topk=None):
        """쿼리와 유사한 페이지 검색 (MaxSim 기반 re-ranking)"""
        if topk is None:
            topk = TOPK_PAGES
            
        self._ensure_indexed(url)
        q_emb = self.embed_query(query)

        client = self.vdb.client
        coll_name = _get_collection_name(url)
        col = client.get_or_create_collection(coll_name)
        
        count = col.count()
        if count == 0:
            return []

        groups = defaultdict(list)
        for q_tok in q_emb:
            res = col.query(
                query_embeddings=[q_tok.numpy().tolist()],
                n_results=min(PER_TOKEN_CANDIDATES, count),
                include=["embeddings", "metadatas", "documents"],
            )
            for e, m, d in zip(res["embeddings"][0], res["metadatas"][0], res["documents"][0]):
                groups[(m["doc_id"], m["page"], d)].append(torch.tensor(e, dtype=TENSOR_DTYPE))

        # MaxSim 점수 계산
        scored = []
        for (doc_id, page, img_path), toks in groups.items():
            page_emb = torch.stack(toks, dim=0).to(dtype=TENSOR_DTYPE)
            q_f = q_emb.to(dtype=TENSOR_DTYPE)

            score = self.processor.score_multi_vector(
                q_f.unsqueeze(0),
                page_emb.unsqueeze(0),
                device="cpu",
            )[0][0].item()

            scored.append({"doc_id": doc_id, "page": page, "image": img_path, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:topk]

    async def generate(self, url, question, *, topk=3, vlm_name=VLM_NAME,
                       temperature=0.2, top_p=0.9, max_new_tokens=256):
        """검색된 페이지 이미지를 VLM에 입력해 답변 생성"""
        hits = await self.retrieve(url, question, topk=topk * 3)
        
        if not hits:
            return {
                "answer": "No relevant pages were retrieved.",
                "used_pages": [],
                "images": [],
                "model": vlm_name,
            }

        sel = hits[:max(1, min(topk, len(hits)))]
        used_pages = [h["page"] + 1 for h in sel]
        img_paths = [h["image"] for h in sel]
        imgs = [self._resize_for_vlm(Image.open(p).convert("RGB")) for p in img_paths]

        vlm, vlm_proc = self._ensure_vlm(vlm_name)

        messages = [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": (
                        "You are a helpful assistant that analyzes document page images and answers questions.\n"
                        "Carefully examine all provided images and answer based on what you see.\n"
                        "If you find relevant information, provide a detailed answer."
                    ),
                }],
            },
            {
                "role": "user",
                "content": (
                    [{"type": "image"} for _ in imgs] + [{
                        "type": "text",
                        "text": (
                            f"Question: {question}\n\n"
                            f"These are pages {', '.join([str(p) for p in used_pages])} from the document.\n"
                            f"Please analyze the images carefully and answer the question."
                        ),
                    }]
                ),
            },
        ]

        chat_text = vlm_proc.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = vlm_proc(
            text=[chat_text],
            images=[imgs],
            padding=True,
            return_tensors="pt",
        ).to(vlm.device)

        with torch.no_grad():
            out = vlm.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_new_tokens=max_new_tokens,
                eos_token_id=vlm.generation_config.eos_token_id
                    if hasattr(vlm, "generation_config") and vlm.generation_config.eos_token_id is not None
                    else None,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        gen_ids = out[:, prompt_len:]
        raw = vlm_proc.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        answer = self._postfix(raw)
        
        # 생성 완료 후 이미지 파일 삭제 (임베딩은 Chroma에 보존)
        # self._cleanup_images(url) 일단 삭제 배제하고, 추후 용량 최적화 방법 고려해보기기

        return {
            "answer": answer,
            "used_pages": used_pages,
            "images": img_paths,
            "model": vlm_name,
        }

    def _cleanup_images(self, url):
        """생성 완료 후 이미지 파일과 Chroma 컬렉션 모두 삭제"""
        try:
            pdf_basename = _get_collection_name(url)
            
            # 1. 이미지 파일 삭제
            cache_dir = os.getenv("COLPALI_IMAGE_CACHE", "./colpali_images")
            out_dir = os.path.join(cache_dir, pdf_basename)
            if os.path.exists(out_dir):
                import shutil
                shutil.rmtree(out_dir)
            
            # 2. Chroma 컬렉션 삭제 (다음 요청에서 재색인되도록)
            client = self.vdb.client
            coll_name = _get_collection_name(url)
            try:
                client.delete_collection(name=coll_name)
            except Exception:
                pass  # 컬렉션이 없어도 무시
                
        except Exception:
            pass  # 삭제 실패해도 서비스 영향 없음

    @staticmethod
    def _resize_for_vlm(img, max_side=1280):
        """VLM 입력용 이미지 리사이즈"""
        w, h = img.size
        if max(w, h) <= max_side:
            return img
        r = max_side / max(w, h)
        return img.resize((int(w * r), int(h * r)))

    @staticmethod
    def _ensure_vlm(vlm_name=VLM_NAME):
        """VLM 모델 로드 (캐싱)"""
        if not hasattr(Colpali, "_vlm_cache"):
            Colpali._vlm_cache = {}
        cache = Colpali._vlm_cache
        if vlm_name in cache:
            return cache[vlm_name]["model"], cache[vlm_name]["proc"]

        model = AutoModelForVision2Seq.from_pretrained(
            vlm_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={0: "24GiB", "cpu": "30GiB"},
        ).eval()
        proc = AutoProcessor.from_pretrained(vlm_name)
        cache[vlm_name] = {"model": model, "proc": proc}
        return model, proc

    @staticmethod
    def _postfix(text: str) -> str:
        """VLM 출력 후처리 (role 태그 제거)"""
        s = text
        for tag in ["system", "user", "assistant"]:
            s = s.replace(f"\n{tag}\n", " ").replace(f"{tag}:\n", " ").replace(f"{tag}:", " ")
        s = "\n".join([ln.strip() for ln in s.splitlines() if ln.strip()])
        return s.strip()
