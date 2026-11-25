"""Configuracion centralizada desde variables de entorno."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
WEB_ENV = Path(__file__).resolve().parents[1] / ".env"
if WEB_ENV.exists():
    load_dotenv(WEB_ENV)
elif ROOT_ENV.exists():
    load_dotenv(ROOT_ENV)


def _normalizar_db_url(raw: str) -> str:
    # Render/Heroku a veces entregan postgres:// o postgresql:// sin driver psycopg3
    if raw.startswith("postgres://"):
        return raw.replace("postgres://", "postgresql+psycopg://", 1)
    if raw.startswith("postgresql://") and "+psycopg" not in raw:
        return raw.replace("postgresql://", "postgresql+psycopg://", 1)
    return raw


class Settings:
    def __init__(self) -> None:
        self.app_env = os.environ.get("APP_ENV", "production")
        self.app_base_url = os.environ.get("APP_BASE_URL", "").rstrip("/")
        self.jwt_secret = os.environ.get("JWT_SECRET", "dev-jwt-secret-cambiar-en-produccion")
        self.session_secret = os.environ.get("SESSION_SECRET", "dev-session-secret-cambiar-en-produccion")
        self.timezone_offset = int(os.environ.get("TIMEZONE_OFFSET", "-4"))

        raw_db = os.environ.get("DATABASE_URL", "sqlite:///./peaje_local_test.db")
        self.database_url = _normalizar_db_url(raw_db)

        self.inference_url = os.environ.get("INFERENCE_URL", "").rstrip("/")
        self.inference_api_token = os.environ.get("INFERENCE_API_TOKEN", "")
        self.inference_timeout = float(os.environ.get("INFERENCE_TIMEOUT_SECONDS", "25"))

        self.google_client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
        self.google_client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
        self.google_android_client_id = os.environ.get("GOOGLE_ANDROID_CLIENT_ID", "")
        self.google_redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI", "")

        admins_raw = os.environ.get(
            "ADMIN_EMAILS",
            "doc.alvaro.pinto.ma@unifranz.edu.bo,cristopherrodriguez1510@gmail.com",
        )
        self.admin_emails = sorted({
            e.strip().lower()
            for e in (
                admins_raw
                + ",doc.alvaro.pinto.ma@unifranz.edu.bo,cristopherrodriguez1510@gmail.com"
            ).split(",")
            if e.strip()
        })

        self.qwen_base_url = os.environ.get("QWEN_BASE_URL", "https://openrouter.ai/api/v1")
        self.qwen_api_key = os.environ.get("QWEN_API_KEY", "")
        self.qwen_model = os.environ.get("QWEN_MODEL", "google/gemma-4-31b-it:free")


@lru_cache
def get_settings() -> Settings:
    return Settings()
