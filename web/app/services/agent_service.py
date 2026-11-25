"""Agente conversacional con Qwen (via API web) + Function Calling estricto contra BD."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI
from sqlalchemy.orm import Session

from app.config import get_settings
from app.services.toll_service import consultar_placa_bd, llamar_inferencia, normalizar_placa

logger = logging.getLogger("peaje.agente")

SYSTEM_PROMPT = """Eres el Agente Conversacional oficial del Sistema Inteligente de Control de Peaje.
Reglas obligatorias e inviolables:
1. Responde UNICAMENTE con datos reales devueltos por las herramientas (base de datos y OCR). NUNCA inventes placas, fechas, montos, propietarios ni pasos.
2. Toda consulta de placa debe clasificarse explicitamente en uno de los 4 estados del sistema:
   - "Registrada y con paso": indica fecha/hora, caseta, categoria y monto.
   - "Registrada sin paso": indica que esta registrada (categoria/saldo) pero nunca paso por el peaje.
   - "Identificada no registrada": indica que la camara la detecto pero no esta registrada (posible evasion).
   - "No existe": indica que no hay ningun registro en el sistema.
3. Si la herramienta devuelve `sugerencias` (por posibles errores OCR como O<->0, I<->1, B<->8, S<->5), preguntale claramente al usuario "¿Quisiste decir <PLACA>?" mostrando las sugerencias.
4. Responde de forma concisa, clara y en español."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "consultar_placa_sistema",
            "description": "Normaliza una placa y consulta su estado exacto, vehiculo registrado, historial de pasos y coincidencias OCR cercanas en la base de datos del peaje.",
            "parameters": {
                "type": "object",
                "properties": {
                    "placa": {
                        "type": "string",
                        "description": "Texto de la placa vehicular a consultar (puede venir con guiones o espacios).",
                    }
                },
                "required": ["placa"],
            },
        },
    }
]


def _extraer_placa_de_texto(texto: str) -> str | None:
    # Busca patrones tipo 2481-KLM, 2481KLM, 412 XYZ o tokens alfanumericos de 5-8 chars con digitos y letras
    candidatos = re.findall(r"\b[A-Za-z0-9]{2,5}[-\s]?[A-Za-z0-9]{2,5}\b", texto or "")
    for cand in candidatos:
        norm = normalizar_placa(cand)
        if 4 <= len(norm) <= 8 and any(c.isdigit() for c in norm) and any(c.isalpha() for c in norm):
            return norm
    return None


def _formatear_respuesta_deterministica(consulta: dict[str, Any], origen_imagen: bool = False) -> str:
    prefijo = f"Placa detectada en la imagen: **{consulta['placa_normalizada']}**\n\n" if origen_imagen else ""
    estado = consulta["estado_sistema"]
    msg = consulta["mensaje"]
    return f"{prefijo}**Estado: {estado}**\n{msg}"


async def ejecutar_agente(
    db: Session,
    mensaje: str = "",
    imagen_bytes: bytes | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    consulta_directa: dict[str, Any] | None = None
    ocr_info: dict[str, Any] | None = None

    # 1. Si viene imagen, primero extraemos la placa con M1 + OCR en INFERENCE_URL
    if imagen_bytes is not None:
        inf = await llamar_inferencia(imagen_bytes, solo_placa=True)
        placa_obj = inf.get("placa") or {}
        texto_ocr = normalizar_placa(placa_obj.get("texto") or "")
        ocr_info = {
            "texto_crudo": placa_obj.get("texto_crudo", ""),
            "placa_normalizada": texto_ocr,
            "confianza": placa_obj.get("confianza", 0.0),
            "formato_bolivia": placa_obj.get("formato_bolivia", False),
            "imagen_anotada": inf.get("imagen_anotada"),
        }
        if not texto_ocr or len(texto_ocr) < 4:
            return {
                "respuesta": "No se pudo detectar una placa legible en la imagen enviada. Por favor envia una foto mas cercana, enfocada y con buena iluminacion de la placa.",
                "estado_sistema": None,
                "placa_normalizada": "",
                "ocr": ocr_info,
                "consulta": None,
                "modelo_llm": "regla_ocr_no_legible",
            }
        consulta_directa = consultar_placa_bd(db, texto_ocr)

    # 2. Si no viene imagen pero se detecta una placa directa en el texto, la consultamos para respaldo/contexto
    placa_en_texto = _extraer_placa_de_texto(mensaje) if imagen_bytes is None else None
    if consulta_directa is None and placa_en_texto:
        consulta_directa = consultar_placa_bd(db, placa_en_texto)

    # 3. Llamada a Qwen via API Web (con Function Calling) si hay API key configurada
    if settings.qwen_api_key:
        try:
            client = AsyncOpenAI(
                base_url=settings.qwen_base_url,
                api_key=settings.qwen_api_key,
                timeout=18.0,
            )
            messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
            if imagen_bytes is not None and consulta_directa is not None:
                user_prompt = (
                    f"El usuario envio una imagen de placa. El sistema OCR detecto '{consulta_directa['placa_normalizada']}' "
                    f"(confianza {ocr_info['confianza']}). Resultado real de la base de datos:\n"
                    f"{json.dumps(consulta_directa, ensure_ascii=False)}\n"
                    f"Mensaje adicional del usuario: {mensaje or '(ninguno)'}"
                )
                messages.append({"role": "user", "content": user_prompt})
                comp = await client.chat.completions.create(
                    model=settings.qwen_model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=450,
                )
                texto_resp = comp.choices[0].message.content or _formatear_respuesta_deterministica(consulta_directa, True)
                return {
                    "respuesta": texto_resp,
                    "estado_sistema": consulta_directa["estado_sistema"],
                    "placa_normalizada": consulta_directa["placa_normalizada"],
                    "ocr": ocr_info,
                    "consulta": consulta_directa,
                    "modelo_llm": settings.qwen_model,
                }

            messages.append({"role": "user", "content": mensaje or "Hola"})
            comp = await client.chat.completions.create(
                model=settings.qwen_model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=450,
            )
            choice_msg = comp.choices[0].message
            tool_calls = getattr(choice_msg, "tool_calls", None) or []

            if tool_calls:
                messages.append(choice_msg.model_dump())
                for tc in tool_calls:
                    if tc.function.name == "consultar_placa_sistema":
                        args = json.loads(tc.function.arguments or "{}")
                        consulta_directa = consultar_placa_bd(db, args.get("placa", ""))
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(consulta_directa, ensure_ascii=False),
                        })
                comp2 = await client.chat.completions.create(
                    model=settings.qwen_model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=450,
                )
                texto_resp = comp2.choices[0].message.content or (
                    _formatear_respuesta_deterministica(consulta_directa) if consulta_directa else "Consulta procesada."
                )
            else:
                if consulta_directa is not None:
                    texto_resp = _formatear_respuesta_deterministica(consulta_directa)
                else:
                    texto_resp = choice_msg.content or "Indicame el numero de placa en texto o sube una imagen para verificar su estado."

            return {
                "respuesta": texto_resp,
                "estado_sistema": consulta_directa["estado_sistema"] if consulta_directa else None,
                "placa_normalizada": consulta_directa["placa_normalizada"] if consulta_directa else "",
                "ocr": ocr_info,
                "consulta": consulta_directa,
                "modelo_llm": settings.qwen_model,
            }
        except Exception as exc:
            logger.warning("Fallo consulta a Qwen API, usando respaldo deterministico: %s", exc)

    # 4. Respaldo deterministico (garantiza 100% disponibilidad aunque falte QWEN_API_KEY o haya timeout)
    if consulta_directa is not None:
        return {
            "respuesta": _formatear_respuesta_deterministica(consulta_directa, origen_imagen=imagen_bytes is not None),
            "estado_sistema": consulta_directa["estado_sistema"],
            "placa_normalizada": consulta_directa["placa_normalizada"],
            "ocr": ocr_info,
            "consulta": consulta_directa,
            "modelo_llm": "respaldo-deterministico",
        }

    return {
        "respuesta": (
            "Soy el agente de control de peaje. Escribe una placa (ejemplo: `2481-KLM` o `8888EVD`) "
            "o adjunta una foto de la placa para consultar su estado en tiempo real."
        ),
        "estado_sistema": None,
        "placa_normalizada": "",
        "ocr": ocr_info,
        "consulta": None,
        "modelo_llm": "respaldo-deterministico",
    }
