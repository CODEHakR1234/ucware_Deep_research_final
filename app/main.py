
# app/main.py
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from app.controller import (
    pdf_summary_controller,
    chat_summary_controller,
    feedback_controller,
    pdf_tutorial_controller,
)
print("[DEBUG] main.py 시작됨 - Tutorial 기능 포함", flush=True)

app = FastAPI(title="Multi-Summary API")

# GZip 압축 미들웨어 (응답 크기 축소)
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[               # 프론트 오리진 **정확히** 넣기
        "http://192.168.0.173:3000",
        "http://172.16.10.117:3000",
        "http://localhost:3000",
        "https://cklsfamily.com",
        "https://www.cklsfamily.com",
        "http://cklsfamily.com",
        "http://www.cklsfamily.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],          # OPTIONS 포함
    allow_headers=["*"],
)

# ───────────────────────────────────────────
# REST Endpoints
# ───────────────────────────────────────────
app.include_router(pdf_summary_controller.router)
app.include_router(chat_summary_controller.router)
app.include_router(feedback_controller.router)
app.include_router(pdf_tutorial_controller.router)
