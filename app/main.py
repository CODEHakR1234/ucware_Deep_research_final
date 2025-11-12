
# app/main.py
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.controller import (
    pdf_summary_controller,
    chat_summary_controller,
    feedback_controller,
    pdf_tutorial_controller,
)
print("[DEBUG] main.py 시작됨 - Tutorial 기능 포함", flush=True)

app = FastAPI(title="Multi-Summary API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[               # 프론트 오리진 **정확히** 넣기
        "http://192.168.0.173:3000",
        "http://172.16.10.117:3000",
        "http://localhost:3000",
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
