"""Punto de entrada principal de la aplicacion web y API REST."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from app.auth import configure_oauth
from app.config import get_settings
from app.database import Base, SessionLocal, engine
from app.routers.api_v1 import router as api_v1_router
from app.routers.web_pages import router as web_pages_router
from app.seed import seed_database


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed_database(db)
    finally:
        db.close()
    configure_oauth()
    yield


settings = get_settings()

app = FastAPI(
    title="Sistema Inteligente de Control de Peaje — IAM-811",
    version="1.0.0",
    description="Panel en tiempo real, API REST v1, autenticacion Google OAuth 2.0 y Agente Conversacional Qwen.",
    lifespan=lifespan,
)

app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret,
    same_site="lax",
    https_only=settings.app_env == "production" and not settings.app_base_url.startswith("http://localhost"),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Sistema"])
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(api_v1_router)
app.include_router(web_pages_router)
