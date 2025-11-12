"""
Captioner – 멀티모달 캡션 생성 (vLLM LLaVA | OpenAI 클라우드)
===========================================================
bytes 이미지 리스트 → 캡션 문자열 리스트 (동일 인덱스 매핑)

주의: 이 클래스는 환경변수를 직접 읽지 않는다. 환경 기반 구성은
`app/utils/captioner_factory.py`가 담당하며, 본 클래스는 전달된
파라미터로만 동작한다.
"""

from __future__ import annotations

import asyncio, base64, os
from typing import List, Literal, Optional
import httpx


class Captioner:
    """멀티모달 배치 캡셔닝 헬퍼 (vLLM OpenAI 로컬 | OpenAI 클라우드)."""

    def __init__(
        self,
        *,
        backend: Literal["openai_local", "openai"],
        model: str,
        api_base: Optional[str] = None,
        openai_api_key: str | None = None,
        timeout: int = 30,
        disabled: bool = False,
    ):
        # 구성 저장
        self.backend = backend
        self.model = model
        # 엔드포인트
        if backend == "openai":
            self.endpoint = api_base or "https://api.openai.com/v1"
        else:
            self.endpoint = api_base or "http://localhost:12001/v1"
        self.openai_api_key = openai_api_key or ""
        self.disabled = disabled

        self._cli = httpx.AsyncClient(timeout=timeout)

    # ─────────────────────────────────────────────────────────
    async def caption(
        self,
        images: List[bytes],
        prompt: str | None = None,
        max_tokens: int = 64,
    ) -> List[str]:

        """
        Args
        ----
        images  : PNG/JPEG bytes 리스트
        prompt  : VLM 프롬프트 (기본 1-2 문장 설명 요청)
        max_tokens : 캡션 최대 토큰

        Returns
        -------
        List[str] : 이미지 순서를 보존한 캡션 문자열 리스트
        """
        if not images:
            return []

        # 테스트용 캡셔닝 비활성화
        if self.disabled:
            print(f"[Captioner] ⚠️  캡셔닝 비활성화됨 (DISABLE_CAPTIONING=true)", flush=True)
            print(f"[Captioner]    기본 캡션 사용: {len(images)}개 이미지", flush=True)
            return ["이미지" for _ in images]

        # 캡셔닝 시도 (OpenAI Chat Completions 포맷)
        print(f"[Captioner] 🔄 VLM 호출 시작:", flush=True)
        print(f"[Captioner]    Backend: {self.backend}", flush=True)
        print(f"[Captioner]    Model: {self.model}", flush=True)
        print(f"[Captioner]    Endpoint: {self.endpoint}/chat/completions", flush=True)
        print(f"[Captioner]    이미지 개수: {len(images)}개", flush=True)
        
        try:
            prompt = prompt or "Describe this image in 1-2 sentences."

            async def _gen_one(img: bytes) -> str:
                b64 = base64.b64encode(img).decode()
                if self.backend == "openai":
                    # OpenAI 클라우드: Authorization 헤더 필요
                    headers = {"Authorization": f"Bearer {self.openai_api_key}"}
                    payload = {
                        "model": self.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                                ],
                            }
                        ],
                        "max_tokens": max_tokens,
                    }
                    r = await self._cli.post(f"{self.endpoint}/chat/completions", json=payload, headers=headers)
                    r.raise_for_status()
                    data = r.json()
                    return (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                    )
                else:
                    # vLLM OpenAI 로컬
                    payload = {
                        "model": self.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                                ],
                            }
                        ],
                        "max_tokens": max_tokens,
                    }
                    r = await self._cli.post(f"{self.endpoint}/chat/completions", json=payload)
                    r.raise_for_status()
                    data = r.json()
                    return (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                    )

            # 여러 장 이미지를 비동기 병렬 처리
            results = await asyncio.gather(*(_gen_one(b) for b in images))
            
            print(f"[Captioner] ✅ VLM 호출 성공!", flush=True)
            print(f"[Captioner]    생성된 캡션 {len(results)}개", flush=True)
            
            # 샘플 캡션 출력 (처음 2개)
            for i, caption in enumerate(results[:2], 1):
                print(f"[Captioner]    샘플 {i}: \"{caption[:60]}...\"" if len(caption) > 60 else f"[Captioner]    샘플 {i}: \"{caption}\"", flush=True)
            
            return results
            
        except Exception as e:
            print(f"[Captioner] ❌ 캡션 생성 실패!", flush=True)
            print(f"[Captioner]    에러: {e}", flush=True)
            print(f"[Captioner]    기본 캡션 사용: {len(images)}개", flush=True)
            
            # VLM 서버 상태 힌트
            if "Connection" in str(e) or "refused" in str(e).lower():
                print(f"[Captioner]    💡 VLM 서버가 실행 중인지 확인하세요: {self.endpoint}", flush=True)
            
            # 기본 캡션 반환
            return ["이미지" for _ in images]

