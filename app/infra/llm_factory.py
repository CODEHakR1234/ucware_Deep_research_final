# app/infra/llm_factory.py
"""llm_factory.py
LLM 인스턴스( OpenAI / HuggingFace )를 생성하는 헬퍼.

환경 변수
---------
- ``LLM_PROVIDER`` : "openai"(기본) 또는 "hf".
- ``LLM_MODEL_NAME``: 모델 이름(e.g. gpt-4o-mini, mistral-7b, etc.).

서비스·인프라 레이어는 이 함수를 통해서만 LLM 객체를 얻도록 해
API 교체·로컬 서버 전환 등을 한 곳에서 관리할 수 있게 한다.
"""

import os
from langchain_openai import ChatOpenAI
# HuggingFaceHub 사용 시 주석 해제
# from langchain import HuggingFaceHub

# ───────────────────── 설정 값 ─────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")

# ───────────────────── 팩토리 함수 ─────────────────────
def get_llm_instance(temperature: float = 0.5):
    """LLM 인스턴스를 반환한다.

    Args
    ----
    temperature : float, optional
        샘플링 온도. 기본 0.5.

    Returns
    -------
    BaseChatModel
        LangChain 호환 LLM 객체(OpenAI 또는 HF).
    """
    if LLM_PROVIDER == "hf":
        # HuggingFaceHub 사용 예시(필요 시 주석 해제)
        # return HuggingFaceHub(
        #     repo_id=LLM_MODEL_NAME,
        #     model_kwargs={"temperature": temperature},
        # )
        return ChatOpenAI(
            model_name=LLM_MODEL_NAME,
            temperature=temperature,
            max_tokens=1000,
            openai_api_base=os.getenv("OPENAI_API_BASE", "http://localhost:12000/v1")
        )
    # 기본: OpenAI ChatCompletion API
    return ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=temperature, max_tokens=1000)

