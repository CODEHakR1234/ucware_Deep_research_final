"""
captioner_factory.py
--------------------
Captioner 인스턴스를 생성/공유하는 팩토리.

환경 변수로 백엔드(vLLM OpenAI 로컬 | OpenAI 클라우드)와 모델, 엔드포인트를 제어한다.
서비스 레이어는 이 팩토리 함수를 통해 캡셔너를 획득하도록 통일한다.
"""

from __future__ import annotations

from typing import Optional
import os
from app.vision.captioner import Captioner

_instance: Optional[Captioner] = None


def get_captioner_instance(*, timeout: int | None = None) -> Captioner:
    """Captioner 싱글턴을 반환한다.

    Args:
        timeout: 초 단위 타임아웃(옵션). 지정 시 최초 생성에만 반영.

    Returns:
        Captioner: 구성된 캡셔너 인스턴스
    """
    global _instance
    if _instance is None:
        backend = os.getenv("CAPTION_BACKEND", "openai_local").lower()
        disabled = os.getenv("DISABLE_CAPTIONING", "false").lower() == "true"
        timeout_val = int(os.getenv("CAPTION_TIMEOUT", timeout or 30))

        if backend == "openai":
            _instance = Captioner(
                backend="openai",
                model=os.getenv("CAPTION_OPENAI_MODEL", "gpt-4o-mini"),
                api_base=os.getenv("CAPTION_API_BASE", "https://api.openai.com/v1"),
                openai_api_key=os.getenv("OPENAI_API_KEY", ""),
                timeout=timeout_val,
                disabled=disabled,
            )
        else:
            _instance = Captioner(
                backend="openai_local",
                model=os.getenv("CAPTION_OPENAI_MODEL", "llava-hf/llava-1.6-7b-hf"),
                api_base=os.getenv("CAPTION_API_BASE", "http://localhost:12001/v1"),
                openai_api_key=None,
                timeout=timeout_val,
                disabled=disabled,
            )
    return _instance


