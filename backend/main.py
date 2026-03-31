"""
Plant Disease Classification API
---------------------------------
Entry point.  Run with:
    uvicorn main:app --reload

Interactive docs: http://127.0.0.1:8000/docs
"""
import sys
import os

# Ensure the backend/ directory is on sys.path so all internal imports resolve
# correctly regardless of the working directory from which uvicorn is launched.
sys.path.insert(0, os.path.dirname(__file__)) #backend/ 加到 sys.path，这样无论从哪个目录用 uvicorn 启动，内部的 config、routes 等包内导入都能正常解析。

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import ALLOWED_ORIGINS
from routes import auth, health, predict
#启动时调用 load_model()，只加载一次模型；关停时预留了清理逻辑（例如释放 GPU
from services.model_service import load_model


# ── Lifespan: load model once at startup ─────────────────────────────────────
#生命周期管理器，启动时调用 load_model()，只加载一次模型；关停时预留了清理逻辑（例如释放 GPU
#yield 把函数分成两部分：yield 之前 → 初始化（startup） yield 之后 → 清理（shutdown）
@asynccontextmanager #is used to define a lifecycle context in FastAPI, allowing initialization before the application starts and cleanup after it shuts down.
async def lifespan(app: FastAPI):
    load_model()
    yield 
    #后面的代码（你这里暂时没写）
    # Add any cleanup here if needed (e.g., release GPU memory)


# ── App factory ──────────────────────────────────────────────────────────────
#创建 FastAPI 应用实例，设置生命周期管理器
app = FastAPI(
    title="Plant Disease Classification API",
    description=(
        "Upload a plant leaf image to get a disease prediction "
        "with a Grad-CAM visual explanation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# Allow the frontend (any origin in dev, restrict in production) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS, #允许的源（origin）列表，* 表示允许所有源（在开发时允许所有域访问，生产时需要限制）
    allow_credentials=True, #允许发送 cookies、HTTP 认证信息等凭证
    allow_methods=["*"], #允许的 HTTP 方法，* 表示允许所有方法
    allow_headers=["*"], #允许的请求头，* 表示允许所有头
)

# ── Routes ───────────────────────────────────────────────────────────────────
app.include_router(health.router,  tags=["Health"])
app.include_router(auth.router,    tags=["Auth"])
app.include_router(predict.router, tags=["Prediction"])
