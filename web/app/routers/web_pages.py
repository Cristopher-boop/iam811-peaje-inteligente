"""Rutas HTML (OAuth 2.0 Google, Panel tiempo real, CRUD, Pasos, Alertas, Chat, Estado)."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.auth import buscar_usuario_autorizado, configure_oauth, get_web_session_user
from app.config import get_settings
from app.database import get_db
from app.models import Alerta, Caseta, Paso, Tarifa, Usuario, Vehiculo
from app.services.toll_service import (
    obtener_resumen_hoy,
    serializar_alerta,
    serializar_paso,
    verificar_salud_inferencia,
)

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter(tags=["Web UI"])


def _require_web_user(request: Request, db: Session) -> Usuario | RedirectResponse:
    user = get_web_session_user(request, db)
    if user is None:
        return RedirectResponse(url="/login", status_code=302)
    return user


def _ctx(request: Request, user: Usuario, pagina: str, **extra) -> dict:
    return {"request": request, "user": user, "pagina": pagina, **extra}


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request, db: Session = Depends(get_db)):
    user = get_web_session_user(request, db)
    if user is not None:
        return RedirectResponse(url="/", status_code=302)
    settings = get_settings()
    oauth_listo = bool(settings.google_client_id and settings.google_client_secret)
    return templates.TemplateResponse(
        request,
        "login.html",
        {"oauth_listo": oauth_listo, "error": request.query_params.get("error")},
    )


@router.get("/auth/google/login")
async def google_login(request: Request):
    settings = get_settings()
    oauth = configure_oauth()
    if "google" not in oauth._clients:
        return RedirectResponse(url="/login?error=oauth_no_configurado", status_code=302)
    redirect_uri = settings.google_redirect_uri or str(request.url_for("google_callback"))
    if settings.app_env == "production" and redirect_uri.startswith("http://") and "localhost" not in redirect_uri:
        redirect_uri = "https://" + redirect_uri[len("http://") :]
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/auth/google/callback", name="google_callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
    oauth = configure_oauth()
    if "google" not in oauth._clients:
        return RedirectResponse(url="/login?error=oauth_no_configurado", status_code=302)
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception:
        return RedirectResponse(url="/login?error=oauth_fallo", status_code=302)

    userinfo = token.get("userinfo")
    if not userinfo:
        userinfo = await oauth.google.userinfo(token=token)
    email = str(userinfo.get("email", "")).strip().lower()
    nombre = str(userinfo.get("name") or email.split("@")[0])

    user = buscar_usuario_autorizado(db, email)
    if user is None or not user.activo:
        request.session.clear()
        return templates.TemplateResponse(
            request,
            "acceso_denegado.html",
            {"email": email, "motivo": "Tu cuenta de Google no esta autorizada en la lista blanca del sistema."},
            status_code=403,
        )

    if nombre and not user.nombre.startswith("Docente") and user.nombre != nombre:
        user.nombre = nombre[:150]
        db.commit()

    request.session["user_id"] = user.id
    return RedirectResponse(url="/", status_code=302)


@router.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)


@router.get("/", response_class=HTMLResponse)
def dashboard_page(request: Request, db: Session = Depends(get_db)):
    res = _require_web_user(request, db)
    if isinstance(res, RedirectResponse):
        return res
    pasos = db.scalars(select(Paso).order_by(desc(Paso.created_at)).limit(25)).all()
    alertas = db.scalars(select(Alerta).where(Alerta.atendida == False).order_by(desc(Alerta.created_at)).limit(10)).all()  # noqa: E712
    casetas = db.scalars(select(Caseta).where(Caseta.activa == True).order_by(Caseta.id)).all()  # noqa: E712
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        _ctx(
            request,
            res,
            "panel",
            resumen=obtener_resumen_hoy(db),
            pasos=[serializar_paso(p) for p in pasos],
            alertas=[serializar_alerta(a) for a in alertas],
            casetas=casetas,
        ),
    )


@router.get("/vehiculos", response_class=HTMLResponse)
def vehiculos_page(request: Request, db: Session = Depends(get_db)):
    res = _require_web_user(request, db)
    if isinstance(res, RedirectResponse):
        return res
    vehiculos = db.scalars(select(Vehiculo).order_by(Vehiculo.placa)).all()
    tarifas = db.scalars(select(Tarifa).order_by(Tarifa.categoria, Tarifa.ejes)).all()
    casetas = db.scalars(select(Caseta).order_by(Caseta.id)).all()
    usuarios = db.scalars(select(Usuario).order_by(Usuario.id)).all() if res.rol == "admin" else []
    return templates.TemplateResponse(
        request,
        "vehiculos.html",
        _ctx(request, res, "vehiculos", vehiculos=vehiculos, tarifas=tarifas, casetas=casetas, usuarios=usuarios),
    )


@router.get("/pasos", response_class=HTMLResponse)
def pasos_page(request: Request, db: Session = Depends(get_db)):
    res = _require_web_user(request, db)
    if isinstance(res, RedirectResponse):
        return res
    pasos = db.scalars(select(Paso).order_by(desc(Paso.created_at)).limit(100)).all()
    return templates.TemplateResponse(
        request,
        "pasos.html",
        _ctx(request, res, "pasos", pasos=[serializar_paso(p) for p in pasos]),
    )


@router.get("/alertas", response_class=HTMLResponse)
def alertas_page(request: Request, db: Session = Depends(get_db)):
    res = _require_web_user(request, db)
    if isinstance(res, RedirectResponse):
        return res
    alertas = db.scalars(select(Alerta).order_by(desc(Alerta.created_at)).limit(100)).all()
    return templates.TemplateResponse(
        request,
        "alertas.html",
        _ctx(request, res, "alertas", alertas=[serializar_alerta(a) for a in alertas]),
    )


@router.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request, db: Session = Depends(get_db)):
    res = _require_web_user(request, db)
    if isinstance(res, RedirectResponse):
        return res
    return templates.TemplateResponse(request, "chat.html", _ctx(request, res, "chat"))


@router.get("/estado", response_class=HTMLResponse)
async def estado_page(request: Request, db: Session = Depends(get_db)):
    res = _require_web_user(request, db)
    if isinstance(res, RedirectResponse):
        return res
    salud = await verificar_salud_inferencia()
    settings = get_settings()
    return templates.TemplateResponse(
        request,
        "estado.html",
        _ctx(
            request,
            res,
            "estado",
            salud=salud,
            inference_url=settings.inference_url or "(no configurada)",
            qwen_model=settings.qwen_model,
            qwen_configurado=bool(settings.qwen_api_key),
        ),
    )
