"""
POST /health → simple health endpoint check.
"""

# ╔════════════════════════════════════════════════════════════╗
# ║ 🚚 IMPORTS
# ╚════════════════════════════════════════════════════════════╝
from core.config import settings

# ╔════════════════════════════════════════════════════════════╗
# ║ 🌐 API
# ╚════════════════════════════════════════════════════════════╝
from fastapi import FastAPI
from contextlib import asynccontextmanager
from tf_keras.models import load_model


# ═════════════════════ MODEL LOADING ═════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model(settings.MODEL_PATH)
    yield


app = FastAPI(
    lifespan=lifespan,
    title=settings.APP_NAME,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ╔════════════════════════════════════════════════════════════╗
# ║ 🥷 MIDDLEWARES
# ╚════════════════════════════════════════════════════════════╝
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# ╔════════════════════════════════════════════════════════════╗
# ║ 🛣️ ROUTERS
# ╚════════════════════════════════════════════════════════════╝
from api.models import (
    text_predict,
    # text_predict_batch,
    text_upload_csv,
)

app.include_router(text_predict)
# app.include_router(text_predict_batch)
app.include_router(text_upload_csv)


# ╔════════════════════════════════════════════════════════════╗
# ║ ⛑️ HEALTH CHECK
# ╚════════════════════════════════════════════════════════════╝
@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "OK",
        "app": settings.APP_NAME,
        "Description": settings.DESCRIPTION,
    }
