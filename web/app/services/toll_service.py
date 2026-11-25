"""Logica de inferencia remota, normalizacion, tarifas, pasos, alertas y consultas."""
from __future__ import annotations

import base64
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from fastapi import HTTPException
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Alerta, Caseta, Evidencia, Paso, Tarifa, Usuario, Vehiculo

FORMATO_BOLIVIA = re.compile(r"^\d{3,4}[A-Z]{3}$")
FORMATO_GENERAL = re.compile(r"^[A-Z0-9]{4,9}$")

CONFUSIONES_OCR = {
    "O": "0",
    "0": "O",
    "I": "1",
    "1": "I",
    "B": "8",
    "8": "B",
    "S": "5",
    "5": "S",
}


def normalizar_placa(texto: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (texto or "").upper())


def validar_formato_placa(placa_norm: str) -> dict[str, bool]:
    return {
        "valida": bool(FORMATO_GENERAL.match(placa_norm)),
        "formato_bolivia": bool(FORMATO_BOLIVIA.match(placa_norm)),
    }


def distancia_levenshtein(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        ndp = [i]
        for j, cb in enumerate(b, 1):
            ndp.append(min(ndp[-1] + 1, dp[j] + 1, dp[j - 1] + (0 if ca == cb else 1)))
        dp = ndp
    return dp[-1]


def buscar_coincidencias_cercanas(db: Session, placa_norm: str) -> list[dict[str, Any]]:
    """Busca placas parecidas por errores tipicos OCR (O<->0, I<->1, B<->8, S<->5) o Levenshtein<=1."""
    if not placa_norm or len(placa_norm) < 4:
        return []

    candidatos_bd: dict[str, str] = {}
    for v in db.scalars(select(Vehiculo)).all():
        candidatos_bd[v.placa] = "vehiculos"
    for p_txt in db.scalars(select(Paso.placa_normalizada).distinct()).all():
        if p_txt and p_txt not in candidatos_bd:
            candidatos_bd[p_txt] = "pasos"

    # Variantes directas de sustitucion OCR
    variantes_ocr: set[str] = set()
    for idx, ch in enumerate(placa_norm):
        if ch in CONFUSIONES_OCR:
            variantes_ocr.add(placa_norm[:idx] + CONFUSIONES_OCR[ch] + placa_norm[idx + 1 :])

    sugerencias: list[dict[str, Any]] = []
    for candidata, origen in candidatos_bd.items():
        if candidata == placa_norm:
            continue
        dist = distancia_levenshtein(placa_norm, candidata)
        es_confusion = candidata in variantes_ocr or _equivalente_ocr(placa_norm, candidata)
        if es_confusion or dist <= 1:
            sugerencias.append({
                "placa": candidata,
                "origen": origen,
                "motivo": "confusion_ocr" if es_confusion else "similitud_cercana",
                "distancia": dist,
            })

    sugerencias.sort(key=lambda s: (s["motivo"] != "confusion_ocr", s["distancia"], s["placa"]))
    return sugerencias[:5]


def _equivalente_ocr(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    diffs = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            if CONFUSIONES_OCR.get(ca) != cb:
                return False
            diffs += 1
    return 1 <= diffs <= 2


async def llamar_inferencia(imagen_bytes: bytes, solo_placa: bool = False) -> dict[str, Any]:
    settings = get_settings()
    if not settings.inference_url:
        raise HTTPException(status_code=503, detail="INFERENCE_URL no configurado en el servidor web")

    endpoint = f"{settings.inference_url}/predict"
    params = {"solo_placa": "true" if solo_placa else "false", "incluir_imagen": "true"}
    headers = {"X-API-Token": settings.inference_api_token} if settings.inference_api_token else {}

    try:
        async with httpx.AsyncClient(timeout=settings.inference_timeout) as client:
            resp = await client.post(
                endpoint,
                params=params,
                headers=headers,
                files={"file": ("frame.jpg", imagen_bytes, "image/jpeg")},
            )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"No se pudo conectar al endpoint de inferencia: {exc}") from exc

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Error en inferencia ({resp.status_code}): {resp.text[:300]}")
    return resp.json()


async def verificar_salud_inferencia() -> dict[str, Any]:
    settings = get_settings()
    if not settings.inference_url:
        return {"ok": False, "url": None, "detalle": "INFERENCE_URL no configurado"}

    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(f"{settings.inference_url}/health")
            return {
                "ok": resp.status_code == 200,
                "status_code": resp.status_code,
                "url": settings.inference_url,
                "latencia_ms": round(resp.elapsed.total_seconds() * 1000, 1),
                "detalle": resp.json() if resp.status_code == 200 else resp.text[:200],
            }
    except Exception as exc:
        return {"ok": False, "url": settings.inference_url, "detalle": str(exc)}


def calcular_tarifa(db: Session, categoria: str | None, ejes: int | None) -> float:
    cat = (categoria or "auto").strip().lower()
    tarifas = db.scalars(select(Tarifa).where(Tarifa.categoria == cat).order_by(Tarifa.ejes)).all()
    if not tarifas:
        return 5.0
    if ejes is None:
        return float(tarifas[0].monto)
    exacta = next((t for t in tarifas if t.ejes == ejes), None)
    if exacta is not None:
        return float(exacta.monto)
    # Si hay mas ejes que la mayor tarifa configurada, toma la mayor tarifa de esa categoria
    menores = [t for t in tarifas if t.ejes <= ejes]
    return float(menores[-1].monto if menores else tarifas[0].monto)


def _decodificar_evidencia(inferencia_res: dict[str, Any], raw_bytes: bytes) -> bytes:
    b64 = inferencia_res.get("imagen_anotada") or ""
    if b64:
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        try:
            return base64.b64decode(b64)
        except Exception:
            pass
    return raw_bytes


def registrar_paso(
    db: Session,
    usuario: Usuario,
    caseta_id: int,
    inferencia_res: dict[str, Any],
    raw_bytes: bytes,
) -> tuple[Paso, Alerta | None, Vehiculo | None]:
    caseta = db.get(Caseta, caseta_id)
    if caseta is None:
        caseta = db.scalar(select(Caseta).order_by(Caseta.id))
    if caseta is None:
        raise HTTPException(status_code=400, detail="No existe ninguna caseta configurada")

    placa_obj = inferencia_res.get("placa") or {}
    placa_raw = placa_obj.get("texto_crudo") or placa_obj.get("texto") or ""
    placa_norm = normalizar_placa(placa_obj.get("texto") or placa_raw)
    conf_ocr = placa_obj.get("confianza")

    vehiculo_obj = inferencia_res.get("vehiculo") or {}
    categoria = vehiculo_obj.get("categoria") or inferencia_res.get("categoria") or "auto"
    ejes = inferencia_res.get("ejes")
    lat_ms = (inferencia_res.get("latencia_ms") or {}).get("total")

    evidencia_bytes = _decodificar_evidencia(inferencia_res, raw_bytes)
    evidencia = Evidencia(contenido=evidencia_bytes, mime_type="image/jpeg")
    db.add(evidencia)
    db.flush()

    vehiculo = db.scalar(select(Vehiculo).where(Vehiculo.placa == placa_norm)) if placa_norm else None
    ejes_calculo = ejes or (vehiculo.ejes if vehiculo else 2)
    monto = 0.0 if not placa_norm else calcular_tarifa(db, categoria, ejes_calculo)

    tipo_alerta: str | None = None
    msg_alerta: str = ""
    if not placa_norm or len(placa_norm) < 4:
        estado = "ILEGIBLE"
        monto = 0.0
        tipo_alerta = "PLACA_ILEGIBLE"
        msg_alerta = f"Vehiculo ({categoria}) detectado en {caseta.nombre} sin placa legible."
    elif vehiculo is None:
        estado = "NO_REGISTRADA"
        tipo_alerta = "PLACA_NO_REGISTRADA"
        msg_alerta = f"Placa {placa_norm} ({categoria}) no esta registrada en el sistema (posible evasion)."
    elif not vehiculo.activo:
        estado = "INACTIVO"
        tipo_alerta = "VEHICULO_INACTIVO"
        msg_alerta = f"Vehiculo {placa_norm} se encuentra inactivo en el sistema."
    elif vehiculo.saldo < monto:
        estado = "SIN_SALDO"
        tipo_alerta = "SIN_SALDO"
        msg_alerta = f"Placa {placa_norm} sin saldo suficiente (saldo: {vehiculo.saldo:.2f} Bs, tarifa: {monto:.2f} Bs)."
    else:
        estado = "AUTORIZADO"
        vehiculo.saldo = round(vehiculo.saldo - monto, 2)

    paso = Paso(
        placa_leida=placa_raw or "(sin lectura)",
        placa_normalizada=placa_norm,
        confianza_ocr=conf_ocr,
        categoria_detectada=categoria,
        ejes_detectados=ejes,
        monto=monto,
        caseta_id=caseta.id,
        usuario_id=usuario.id,
        imagen_url=f"/api/v1/evidencias/{evidencia.id}.jpg",
        estado=estado,
        latencia_ms=lat_ms,
    )
    db.add(paso)
    db.flush()

    alerta: Alerta | None = None
    if tipo_alerta is not None:
        alerta = Alerta(
            paso_id=paso.id,
            tipo=tipo_alerta,
            mensaje=msg_alerta,
            atendida=False,
        )
        db.add(alerta)

    db.commit()
    db.refresh(paso)
    if alerta is not None:
        db.refresh(alerta)
    return paso, alerta, vehiculo


def consultar_placa_bd(db: Session, placa_raw: str) -> dict[str, Any]:
    """Devuelve exactamente uno de los 4 estados exigidos por la consigna + coincidencias OCR."""
    placa_norm = normalizar_placa(placa_raw)
    formato = validar_formato_placa(placa_norm)

    if not placa_norm:
        return {
            "placa_consultada": placa_raw,
            "placa_normalizada": "",
            "formato": formato,
            "estado_sistema": "No existe",
            "mensaje": "No se proporciono una placa valida para consultar.",
            "sugerencias": [],
        }

    vehiculo = db.scalar(select(Vehiculo).where(Vehiculo.placa == placa_norm))
    pasos = db.scalars(
        select(Paso).where(Paso.placa_normalizada == placa_norm).order_by(desc(Paso.created_at))
    ).all()

    historial = [serializar_paso(p) for p in pasos[:10]]
    ultimo = historial[0] if historial else None

    if vehiculo is not None and pasos:
        estado_sistema = "Registrada y con paso"
        mensaje = (
            f"La placa {placa_norm} esta registrada ({vehiculo.categoria}, propietario: {vehiculo.propietario}, "
            f"saldo: {vehiculo.saldo:.2f} Bs) y tiene {len(pasos)} paso(s). "
            f"Ultimo paso: {ultimo['fecha_local']} en {ultimo['caseta_nombre']}, "
            f"categoria {ultimo['categoria_detectada']}, monto {ultimo['monto']:.2f} Bs ({ultimo['estado']})."
        )
        sugerencias: list[dict[str, Any]] = []
    elif vehiculo is not None and not pasos:
        estado_sistema = "Registrada sin paso"
        mensaje = (
            f"La placa {placa_norm} esta registrada ({vehiculo.categoria}, {vehiculo.ejes} ejes, "
            f"propietario: {vehiculo.propietario}, saldo: {vehiculo.saldo:.2f} Bs), "
            f"pero nunca ha registrado pasos por el peaje."
        )
        sugerencias = []
    elif vehiculo is None and pasos:
        estado_sistema = "Identificada no registrada"
        mensaje = (
            f"La placa {placa_norm} fue detectada por las camaras del peaje en {len(pasos)} ocasion(es) "
            f"(ultimo registro: {ultimo['fecha_local']} en {ultimo['caseta_nombre']}, monto ref. {ultimo['monto']:.2f} Bs), "
            f"pero NO esta registrada en el padron de vehiculos (posible evasion)."
        )
        sugerencias = buscar_coincidencias_cercanas(db, placa_norm)
    else:
        estado_sistema = "No existe"
        sugerencias = buscar_coincidencias_cercanas(db, placa_norm)
        if sugerencias:
            nombres = ", ".join(s["placa"] for s in sugerencias)
            mensaje = f"No existe ningun registro para la placa {placa_norm}. ¿Quisiste decir: {nombres}?"
        else:
            mensaje = f"No existe ningun registro de la placa {placa_norm} ni como vehiculo ni en pasos capturados."

    return {
        "placa_consultada": placa_raw,
        "placa_normalizada": placa_norm,
        "formato": formato,
        "estado_sistema": estado_sistema,
        "mensaje": mensaje,
        "vehiculo": {
            "placa": vehiculo.placa,
            "propietario": vehiculo.propietario,
            "categoria": vehiculo.categoria,
            "ejes": vehiculo.ejes,
            "saldo": round(vehiculo.saldo, 2),
            "activo": vehiculo.activo,
        } if vehiculo else None,
        "total_pasos": len(pasos),
        "ultimo_paso": ultimo,
        "historial": historial,
        "sugerencias": sugerencias,
    }


def _formatear_local(dt: datetime | None) -> str:
    if dt is None:
        return ""
    offset = timedelta(hours=get_settings().timezone_offset)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local_dt = dt.astimezone(timezone(offset))
    return local_dt.strftime("%Y-%m-%d %H:%M:%S")


def serializar_paso(p: Paso) -> dict[str, Any]:
    return {
        "id": p.id,
        "placa_leida": p.placa_leida,
        "placa_normalizada": p.placa_normalizada,
        "confianza_ocr": p.confianza_ocr,
        "categoria_detectada": p.categoria_detectada,
        "ejes_detectados": p.ejes_detectados,
        "monto": round(float(p.monto), 2),
        "caseta_id": p.caseta_id,
        "caseta_nombre": p.caseta.nombre if p.caseta else f"Caseta #{p.caseta_id}",
        "imagen_url": p.imagen_url,
        "estado": p.estado,
        "latencia_ms": p.latencia_ms,
        "created_at": p.created_at.isoformat() if p.created_at else None,
        "fecha_local": _formatear_local(p.created_at),
    }


def serializar_alerta(a: Alerta) -> dict[str, Any]:
    return {
        "id": a.id,
        "paso_id": a.paso_id,
        "tipo": a.tipo,
        "mensaje": a.mensaje,
        "atendida": a.atendida,
        "placa": a.paso.placa_normalizada if a.paso else "",
        "caseta_nombre": a.paso.caseta.nombre if a.paso and a.paso.caseta else "",
        "imagen_url": a.paso.imagen_url if a.paso else None,
        "created_at": a.created_at.isoformat() if a.created_at else None,
        "fecha_local": _formatear_local(a.created_at),
    }


def obtener_resumen_hoy(db: Session, caseta_id: int | None = None) -> dict[str, Any]:
    offset = timedelta(hours=get_settings().timezone_offset)
    tz_local = timezone(offset)
    ahora_local = datetime.now(tz_local)
    inicio_dia_utc = datetime(ahora_local.year, ahora_local.month, ahora_local.day, tzinfo=tz_local).astimezone(timezone.utc)

    q = select(Paso).where(Paso.created_at >= inicio_dia_utc)
    if caseta_id is not None:
        q = q.where(Paso.caseta_id == caseta_id)
    pasos_hoy = db.scalars(q).all()

    recaudacion = round(sum(p.monto for p in pasos_hoy if p.estado == "AUTORIZADO"), 2)
    por_categoria: dict[str, int] = {}
    for p in pasos_hoy:
        cat = p.categoria_detectada or "sin_categoria"
        por_categoria[cat] = por_categoria.get(cat, 0) + 1

    alertas_pendientes = db.scalar(select(func.count(Alerta.id)).where(Alerta.atendida == False)) or 0  # noqa: E712

    return {
        "fecha_local": ahora_local.strftime("%Y-%m-%d"),
        "caseta_id": caseta_id,
        "total_pasos_hoy": len(pasos_hoy),
        "recaudacion_hoy": recaudacion,
        "alertas_pendientes": int(alertas_pendientes),
        "por_categoria": por_categoria,
    }
