"""API REST v1 del Sistema de Control de Peaje."""
from __future__ import annotations

import asyncio
import base64
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from sqlalchemy import desc, select
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from app.auth import (
    buscar_usuario_autorizado,
    crear_jwt_sistema,
    decodificar_jwt_sistema,
    get_current_user_api,
    require_admin_api,
    verificar_google_id_token,
)
from app.database import get_db
from app.models import Alerta, Caseta, Evidencia, Paso, Tarifa, Usuario, Vehiculo
from app.realtime import hub
from app.schemas import (
    AgenteChatRequest,
    CasetaCreate,
    GoogleAuthRequest,
    PlacaConsultarRequest,
    RecargaRequest,
    TarifaUpsert,
    UsuarioCreate,
    UsuarioToggle,
    VehiculoCreate,
    VehiculoUpdate,
)
from app.services.agent_service import ejecutar_agente
from app.services.toll_service import (
    consultar_placa_bd,
    llamar_inferencia,
    normalizar_placa,
    obtener_resumen_hoy,
    registrar_paso,
    serializar_alerta,
    serializar_paso,
    verificar_salud_inferencia,
)

router = APIRouter(prefix="/api/v1", tags=["API Peaje v1"])


# ------------------------------------------------------------------------------
# 1. Autenticacion Google (unica ruta publica de API v1)
# ------------------------------------------------------------------------------
@router.post("/auth/google", summary="Autentica un token de Google y devuelve JWT del sistema")
async def auth_google(payload: GoogleAuthRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    token_raw = payload.id_token or payload.token
    if not token_raw:
        raise HTTPException(status_code=400, detail="Debes enviar 'id_token'")

    info = await verificar_google_id_token(token_raw)
    email = str(info["email"]).strip().lower()
    user = buscar_usuario_autorizado(db, email)
    if user is None or not user.activo:
        raise HTTPException(
            status_code=403,
            detail=f"Acceso denegado: la cuenta {email} no esta autorizada en la lista blanca.",
        )

    if info.get("name") and user.nombre.startswith("Docente") is False and user.nombre != info["name"]:
        user.nombre = str(info["name"])[:150]
        db.commit()

    jwt_token = crear_jwt_sistema(user)
    casetas = db.scalars(select(Caseta).where(Caseta.activa == True).order_by(Caseta.id)).all()  # noqa: E712
    return {
        "access_token": jwt_token,
        "token_type": "bearer",
        "usuario": {
            "id": user.id,
            "email_google": user.email_google,
            "nombre": user.nombre,
            "rol": user.rol,
        },
        "casetas": [{"id": c.id, "nombre": c.nombre, "ubicacion": c.ubicacion} for c in casetas],
    }


# ------------------------------------------------------------------------------
# 2. Registro de paso por frame + tiempo real
# ------------------------------------------------------------------------------
@router.post("/pasos/frame", summary="Recibe un frame desde la app movil, infiere, registra paso y emite en tiempo real")
async def registrar_paso_frame(
    request: Request,
    caseta_id: int = Form(default=1),
    file: UploadFile | None = File(default=None),
    imagen: UploadFile | None = File(default=None),
    frame: UploadFile | None = File(default=None),
    user: Usuario = Depends(get_current_user_api),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    archivo = file or imagen or frame
    raw_bytes: bytes | None = None
    if archivo is not None:
        raw_bytes = await archivo.read()
    else:
        ctype = (request.headers.get("content-type") or "").lower()
        if "application/json" in ctype:
            body = await request.json()
            caseta_id = int(body.get("caseta_id", caseta_id))
            b64 = body.get("image_base64") or body.get("imagen_base64") or body.get("image")
            if b64:
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                raw_bytes = base64.b64decode(b64)

    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Debes adjuntar una imagen en 'file', 'imagen' o 'image_base64'")

    inferencia_res = await llamar_inferencia(raw_bytes, solo_placa=False)
    paso, alerta, vehiculo = registrar_paso(db, user, caseta_id, inferencia_res, raw_bytes)
    resumen = obtener_resumen_hoy(db)

    paso_json = serializar_paso(paso)
    alerta_json = serializar_alerta(alerta) if alerta is not None else None

    await hub.broadcast("nuevo_paso", {"paso": paso_json, "resumen": resumen})
    if alerta_json is not None:
        await hub.broadcast("nueva_alerta", {"alerta": alerta_json, "resumen": resumen})

    return {
        "paso": paso_json,
        "alerta": alerta_json,
        "saldo_restante": round(vehiculo.saldo, 2) if vehiculo else None,
        "propietario": vehiculo.propietario if vehiculo else None,
        "inferencia": {
            "placa": inferencia_res.get("placa"),
            "vehiculo": inferencia_res.get("vehiculo"),
            "ejes": inferencia_res.get("ejes"),
            "vista": inferencia_res.get("vista"),
            "keypoints": inferencia_res.get("keypoints", []),
            "latencia_ms": inferencia_res.get("latencia_ms"),
        },
        "resumen_hoy": resumen,
    }


# ------------------------------------------------------------------------------
# 3. Consultas de placas (texto, imagen, historial) y resumen de caseta
# ------------------------------------------------------------------------------
@router.post("/placas/consultar", summary="Consulta el estado de una placa por texto")
def consultar_placa_texto(
    payload: PlacaConsultarRequest,
    _: Usuario = Depends(get_current_user_api),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    return consultar_placa_bd(db, payload.placa)


@router.post("/placas/consultar-imagen", summary="Extrae la placa desde imagen (M1 + OCR) y consulta su estado")
async def consultar_placa_imagen(
    file: UploadFile | None = File(default=None),
    imagen: UploadFile | None = File(default=None),
    _: Usuario = Depends(get_current_user_api),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    archivo = file or imagen
    if archivo is None:
        raise HTTPException(status_code=400, detail="Debes adjuntar una imagen en 'file' o 'imagen'")
    raw = await archivo.read()
    inf = await llamar_inferencia(raw, solo_placa=True)
    placa_obj = inf.get("placa") or {}
    texto = normalizar_placa(placa_obj.get("texto") or "")
    if not texto or len(texto) < 4:
        return {
            "placa_detectada": False,
            "placa_normalizada": "",
            "ocr": placa_obj,
            "imagen_anotada": inf.get("imagen_anotada"),
            "estado_sistema": None,
            "mensaje": "No se pudo leer una placa valida en la imagen. Envia otra foto mas clara.",
        }

    consulta = consultar_placa_bd(db, texto)
    consulta["placa_detectada"] = True
    consulta["ocr"] = placa_obj
    consulta["imagen_anotada"] = inf.get("imagen_anotada")
    return consulta


@router.get("/placas/{placa}/historial", summary="Historial de pasos de una placa")
def historial_placa(
    placa: str,
    _: Usuario = Depends(get_current_user_api),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    return consultar_placa_bd(db, placa)


@router.get("/casetas/{caseta_id}/resumen", summary="Conteo y recaudacion del dia para una caseta")
def resumen_caseta(
    caseta_id: int,
    _: Usuario = Depends(get_current_user_api),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    caseta = db.get(Caseta, caseta_id)
    if caseta is None:
        raise HTTPException(status_code=404, detail="Caseta no encontrada")
    res = obtener_resumen_hoy(db, caseta_id=caseta_id)
    res["caseta_nombre"] = caseta.nombre
    return res


# ------------------------------------------------------------------------------
# 4. Agente conversacional (texto o imagen)
# ------------------------------------------------------------------------------
@router.post("/agente/chat", summary="Envia un mensaje o foto de placa al agente conversacional")
async def agente_chat(
    request: Request,
    mensaje: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    imagen: UploadFile | None = File(default=None),
    _: Usuario = Depends(get_current_user_api),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    img_bytes: bytes | None = None
    msg_texto = mensaje or ""

    archivo = file or imagen
    if archivo is not None:
        img_bytes = await archivo.read()
    else:
        ctype = (request.headers.get("content-type") or "").lower()
        if "application/json" in ctype:
            body = AgenteChatRequest(**(await request.json()))
            msg_texto = body.mensaje
            if body.imagen_base64:
                b64 = body.imagen_base64.split(",", 1)[1] if "," in body.imagen_base64 else body.imagen_base64
                img_bytes = base64.b64decode(b64)

    if not msg_texto.strip() and img_bytes is None:
        raise HTTPException(status_code=400, detail="Debes enviar un mensaje de texto o una imagen")

    return await ejecutar_agente(db, mensaje=msg_texto, imagen_bytes=img_bytes)


# ------------------------------------------------------------------------------
# 5. Tiempo real (SSE + WebSocket), Evidencias y Estado
# ------------------------------------------------------------------------------
@router.get("/stream", summary="Stream Server-Sent Events para actualizaciones en tiempo real")
async def sse_stream(
    request: Request,
    _: Usuario = Depends(get_current_user_api),
):
    queue = hub.subscribe_sse()

    async def event_generator():
        try:
            yield {"event": "connected", "data": '{"status":"ok"}'}
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield {"event": "message", "data": payload}
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "{}"}
        finally:
            hub.unsubscribe_sse(queue)

    return EventSourceResponse(event_generator())


@router.websocket("/ws/panel")
async def ws_panel(websocket: WebSocket, token: str = Query(default=""), db: Session = Depends(get_db)):
    try:
        claims = decodificar_jwt_sistema(token)
        user = db.get(Usuario, int(claims["sub"]))
        if user is None or not user.activo:
            await websocket.close(code=1008)
            return
    except Exception:
        await websocket.close(code=1008)
        return

    await hub.register_ws(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        hub.unregister_ws(websocket)


@router.get("/evidencias/{evidencia_id}.jpg", summary="Imagen de evidencia anotada almacenada en BD")
def obtener_evidencia(
    evidencia_id: int,
    _: Usuario = Depends(get_current_user_api),
    db: Session = Depends(get_db),
) -> Response:
    ev = db.get(Evidencia, evidencia_id)
    if ev is None:
        raise HTTPException(status_code=404, detail="Evidencia no encontrada")
    return Response(content=ev.contenido, media_type=ev.mime_type)


@router.get("/estado-inferencia", summary="Verifica la salud del endpoint de inferencia en la nube")
async def estado_inferencia(_: Usuario = Depends(get_current_user_api)) -> dict[str, Any]:
    return await verificar_salud_inferencia()


# ------------------------------------------------------------------------------
# 6. Listados y CRUD administrativo (Vehiculos, Pasos, Alertas, Tarifas, Casetas, Usuarios)
# ------------------------------------------------------------------------------
@router.get("/pasos")
def listar_pasos(
    limit: int = Query(default=50, ge=1, le=200),
    _: Usuario = Depends(get_current_user_api),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    pasos = db.scalars(select(Paso).order_by(desc(Paso.created_at)).limit(limit)).all()
    return {"items": [serializar_paso(p) for p in pasos], "resumen_hoy": obtener_resumen_hoy(db)}


@router.get("/alertas")
def listar_alertas(
    solo_pendientes: bool = Query(default=False),
    _: Usuario = Depends(get_current_user_api),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    q = select(Alerta).order_by(desc(Alerta.created_at))
    if solo_pendientes:
        q = q.where(Alerta.atendida == False)  # noqa: E712
    alertas = db.scalars(q.limit(100)).all()
    return {"items": [serializar_alerta(a) for a in alertas]}


@router.post("/alertas/{alerta_id}/atender")
async def atender_alerta(
    alerta_id: int,
    _: Usuario = Depends(get_current_user_api),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    alerta = db.get(Alerta, alerta_id)
    if alerta is None:
        raise HTTPException(status_code=404, detail="Alerta no encontrada")
    alerta.atendida = True
    db.commit()
    resumen = obtener_resumen_hoy(db)
    await hub.broadcast("alerta_atendida", {"alerta_id": alerta_id, "resumen": resumen})
    return {"ok": True, "resumen": resumen}


@router.get("/vehiculos")
def listar_vehiculos(_: Usuario = Depends(get_current_user_api), db: Session = Depends(get_db)) -> dict[str, Any]:
    items = db.scalars(select(Vehiculo).order_by(Vehiculo.placa)).all()
    return {
        "items": [
            {
                "id": v.id,
                "placa": v.placa,
                "propietario": v.propietario,
                "categoria": v.categoria,
                "ejes": v.ejes,
                "saldo": round(v.saldo, 2),
                "activo": v.activo,
            }
            for v in items
        ]
    }


@router.post("/vehiculos", status_code=201)
def crear_vehiculo(
    payload: VehiculoCreate,
    _: Usuario = Depends(require_admin_api),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    placa = normalizar_placa(payload.placa)
    if len(placa) < 4:
        raise HTTPException(status_code=400, detail="Placa invalida")
    if db.scalar(select(Vehiculo).where(Vehiculo.placa == placa)) is not None:
        raise HTTPException(status_code=409, detail=f"La placa {placa} ya existe")
    v = Vehiculo(
        placa=placa,
        propietario=payload.propietario.strip(),
        categoria=payload.categoria,
        ejes=payload.ejes,
        saldo=round(payload.saldo, 2),
        activo=payload.activo,
    )
    db.add(v)
    db.commit()
    db.refresh(v)
    return {"id": v.id, "placa": v.placa}


@router.put("/vehiculos/{vehiculo_id}")
def actualizar_vehiculo(
    vehiculo_id: int,
    payload: VehiculoUpdate,
    _: Usuario = Depends(require_admin_api),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    v = db.get(Vehiculo, vehiculo_id)
    if v is None:
        raise HTTPException(status_code=404, detail="Vehiculo no encontrado")
    if payload.propietario is not None:
        v.propietario = payload.propietario.strip()
    if payload.categoria is not None:
        v.categoria = payload.categoria
    if payload.ejes is not None:
        v.ejes = payload.ejes
    if payload.saldo is not None:
        v.saldo = round(payload.saldo, 2)
    if payload.activo is not None:
        v.activo = payload.activo
    db.commit()
    return {"ok": True}


@router.post("/vehiculos/{vehiculo_id}/recargar")
def recargar_vehiculo(
    vehiculo_id: int,
    payload: RecargaRequest,
    _: Usuario = Depends(require_admin_api),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    v = db.get(Vehiculo, vehiculo_id)
    if v is None:
        raise HTTPException(status_code=404, detail="Vehiculo no encontrado")
    v.saldo = round(v.saldo + payload.monto, 2)
    db.commit()
    return {"ok": True, "saldo": v.saldo}


@router.delete("/vehiculos/{vehiculo_id}")
def eliminar_vehiculo(
    vehiculo_id: int,
    _: Usuario = Depends(require_admin_api),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    v = db.get(Vehiculo, vehiculo_id)
    if v is None:
        raise HTTPException(status_code=404, detail="Vehiculo no encontrado")
    db.delete(v)
    db.commit()
    return {"ok": True}


@router.get("/tarifas")
def listar_tarifas(_: Usuario = Depends(get_current_user_api), db: Session = Depends(get_db)) -> dict[str, Any]:
    items = db.scalars(select(Tarifa).order_by(Tarifa.categoria, Tarifa.ejes)).all()
    return {"items": [{"id": t.id, "categoria": t.categoria, "ejes": t.ejes, "monto": t.monto} for t in items]}


@router.post("/tarifas")
def upsert_tarifa(
    payload: TarifaUpsert,
    _: Usuario = Depends(require_admin_api),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    t = db.scalar(select(Tarifa).where(Tarifa.categoria == payload.categoria, Tarifa.ejes == payload.ejes))
    if t is None:
        t = Tarifa(categoria=payload.categoria, ejes=payload.ejes, monto=round(payload.monto, 2))
        db.add(t)
    else:
        t.monto = round(payload.monto, 2)
    db.commit()
    return {"ok": True}


@router.get("/casetas")
def listar_casetas(_: Usuario = Depends(get_current_user_api), db: Session = Depends(get_db)) -> dict[str, Any]:
    items = db.scalars(select(Caseta).order_by(Caseta.id)).all()
    return {"items": [{"id": c.id, "nombre": c.nombre, "ubicacion": c.ubicacion, "activa": c.activa} for c in items]}


@router.post("/casetas", status_code=201)
def crear_caseta(
    payload: CasetaCreate,
    _: Usuario = Depends(require_admin_api),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    c = Caseta(nombre=payload.nombre.strip(), ubicacion=payload.ubicacion.strip(), activa=payload.activa)
    db.add(c)
    db.commit()
    db.refresh(c)
    return {"id": c.id, "nombre": c.nombre}


@router.get("/usuarios")
def listar_usuarios(_: Usuario = Depends(require_admin_api), db: Session = Depends(get_db)) -> dict[str, Any]:
    items = db.scalars(select(Usuario).order_by(Usuario.id)).all()
    return {
        "items": [
            {"id": u.id, "email_google": u.email_google, "nombre": u.nombre, "rol": u.rol, "activo": u.activo}
            for u in items
        ]
    }


@router.post("/usuarios", status_code=201)
def crear_usuario(
    payload: UsuarioCreate,
    _: Usuario = Depends(require_admin_api),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    email = payload.email_google.strip().lower()
    if db.scalar(select(Usuario).where(Usuario.email_google == email)) is not None:
        raise HTTPException(status_code=409, detail="El correo ya existe en la lista blanca")
    u = Usuario(email_google=email, nombre=payload.nombre.strip(), rol=payload.rol, activo=payload.activo)
    db.add(u)
    db.commit()
    db.refresh(u)
    return {"id": u.id, "email_google": u.email_google}


@router.patch("/usuarios/{usuario_id}")
def actualizar_usuario(
    usuario_id: int,
    payload: UsuarioToggle,
    _: Usuario = Depends(require_admin_api),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    u = db.get(Usuario, usuario_id)
    if u is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    if payload.rol is not None:
        u.rol = payload.rol
    if payload.activo is not None:
        u.activo = payload.activo
    db.commit()
    return {"ok": True}
