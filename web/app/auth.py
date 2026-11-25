"""Autenticacion exclusiva con Google OAuth 2.0 / Google Sign-In, JWT y RBAC."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import jwt
from authlib.integrations.starlette_client import OAuth
from fastapi import Depends, HTTPException, Request, status
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import Usuario

oauth = OAuth()


def configure_oauth() -> OAuth:
    settings = get_settings()
    if settings.google_client_id and settings.google_client_secret and "google" not in oauth._clients:
        oauth.register(
            name="google",
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret,
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )
    return oauth


async def verificar_google_id_token(token: str) -> dict[str, Any]:
    """Verifica el id_token enviado por Flutter/Google Sign-In contra Google."""
    settings = get_settings()
    allowed_audiences = [
        cid for cid in (settings.google_client_id, settings.google_android_client_id) if cid
    ]
    if not token:
        raise HTTPException(status_code=400, detail="Token de Google vacio")

    # 1. Verificacion criptografica oficial con google-auth
    try:
        req = google_requests.Request()
        info = google_id_token.verify_oauth2_token(token, req)
        aud = info.get("aud")
        azp = info.get("azp")
        if allowed_audiences and aud not in allowed_audiences and azp not in allowed_audiences:
            raise HTTPException(status_code=401, detail="Client ID de Google no autorizado para este sistema")
        if not info.get("email") or not info.get("email_verified", True):
            raise HTTPException(status_code=401, detail="Correo de Google no verificado")
        return info
    except HTTPException:
        raise
    except Exception:
        pass

    # 2. Respaldo via endpoint tokeninfo de Google (por si llegan access/id tokens de pruebas)
    async with httpx.AsyncClient(timeout=8.0) as client:
        resp = await client.get("https://oauth2.googleapis.com/tokeninfo", params={"id_token": token})
        if resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Token de Google invalido o expirado")
        info = resp.json()
        aud = info.get("aud")
        azp = info.get("azp")
        if allowed_audiences and aud not in allowed_audiences and azp not in allowed_audiences:
            raise HTTPException(status_code=401, detail="Client ID de Google no coincide")
        if not info.get("email"):
            raise HTTPException(status_code=401, detail="El token de Google no incluye email")
        return info


def crear_jwt_sistema(usuario: Usuario) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(usuario.id),
        "email": usuario.email_google,
        "nombre": usuario.nombre,
        "rol": usuario.rol,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=24)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decodificar_jwt_sistema(token: str) -> dict[str, Any]:
    settings = get_settings()
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token de sesion invalido o expirado") from exc


def buscar_usuario_autorizado(db: Session, email: str) -> Usuario | None:
    return db.scalar(select(Usuario).where(Usuario.email_google == email.strip().lower()))


def get_web_session_user(request: Request, db: Session) -> Usuario | None:
    uid = request.session.get("user_id")
    if not uid:
        return None
    user = db.get(Usuario, int(uid))
    if user is None or not user.activo:
        request.session.clear()
        return None
    return user


def get_current_user_api(request: Request, db: Session = Depends(get_db)) -> Usuario:
    """Exige autenticacion via Bearer JWT (movil/API) o sesion web valida."""
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        claims = decodificar_jwt_sistema(token)
        user = db.get(Usuario, int(claims["sub"]))
        if user is None or not user.activo:
            raise HTTPException(status_code=403, detail="Usuario no autorizado o desactivado")
        return user

    user = get_web_session_user(request, db)
    if user is not None:
        return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Autenticacion requerida. Inicia sesion con Google.",
    )


def require_admin_api(user: Usuario = Depends(get_current_user_api)) -> Usuario:
    if user.rol != "admin":
        raise HTTPException(status_code=403, detail="Se requiere rol de administrador")
    return user
