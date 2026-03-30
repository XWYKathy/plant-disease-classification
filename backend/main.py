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
sys.path.insert(0, os.path.dirname(__file__))

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import ALLOWED_ORIGINS
from routes import auth, health, predict
from services.model_service import load_model


# ── Lifespan: load model once at startup ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    # Add any cleanup here if needed (e.g., release GPU memory)


# ── App factory ──────────────────────────────────────────────────────────────

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
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ───────────────────────────────────────────────────────────────────
app.include_router(health.router,  tags=["Health"])
app.include_router(auth.router,    tags=["Auth"])
app.include_router(predict.router, tags=["Prediction"])
