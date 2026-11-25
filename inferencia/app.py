"""Servidor FastAPI de inferencia unificada (M1 + M2 + M3) para RunPod o Colab."""
from __future__ import annotations

import base64
import json
import logging
import os
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from pipeline import TollInferencePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("peaje.api")

API_TOKEN = os.environ.get("API_TOKEN") or os.environ.get("INFERENCE_API_TOKEN") or secrets.token_urlsafe(16)
_pipeline: TollInferencePipeline | None = None


def get_pipeline() -> TollInferencePipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Modelos aun inicializando")
    return _pipeline


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _pipeline
    logger.info("Cargando modelos M1, M2 y M3...")
    _pipeline = TollInferencePipeline()
    pod_id = os.environ.get("RUNPOD_POD_ID")
    puerto = os.environ.get("PORT", "8000")
    if pod_id:
        logger.info("Endpoint RunPod activo: https://%s-%s.proxy.runpod.net", pod_id, puerto)
    logger.info("Token de inferencia activo (longitud=%d)", len(API_TOKEN))
    yield


app = FastAPI(
    title="Peaje AI — Servicio de Inferencia Unificada",
    version="1.0.0",
    description="Ejecuta M1 (Placas + OCR), M2 (Segmentacion YOLO-seg) y M3 (Keypoints de ruedas YOLO-pose).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validar_token(x_api_token: str | None, authorization: str | None) -> None:
    token = x_api_token
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Token de inferencia invalido")


async def _extraer_bytes_imagen(request: Request) -> bytes:
    ctype = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" in ctype:
        form = await request.form()
        archivo = form.get("file") or form.get("imagen") or form.get("image")
        if archivo is None or not hasattr(archivo, "read"):
            raise HTTPException(status_code=400, detail="Falta el campo multipart 'file' o 'imagen'")
        return await archivo.read()

    if "application/json" in ctype:
        payload = await request.json()
        b64 = payload.get("image_base64") or payload.get("imagen_base64") or payload.get("image")
        if not b64:
            raise HTTPException(status_code=400, detail="Falta 'image_base64' en el JSON")
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        return base64.b64decode(b64)

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Cuerpo de imagen vacio")
    return body


@app.get("/health")
def health() -> dict:
    return get_pipeline().estado_salud()


@app.post("/predict")
async def predict(
    request: Request,
    solo_placa: bool = Query(default=False, description="Si es true, ejecuta solo M1 + OCR"),
    incluir_imagen: bool = Query(default=True, description="Si es true, devuelve imagen anotada en base64"),
    x_api_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict:
    _validar_token(x_api_token, authorization)
    datos = await _extraer_bytes_imagen(request)
    try:
        return await run_in_threadpool(get_pipeline().inferir, datos, solo_placa, incluir_imagen)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ocr")
async def ocr_solo_placa(
    request: Request,
    incluir_imagen: bool = Query(default=False),
    x_api_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> dict:
    _validar_token(x_api_token, authorization)
    datos = await _extraer_bytes_imagen(request)
    try:
        return await run_in_threadpool(get_pipeline().inferir, datos, True, incluir_imagen)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.websocket("/ws")
async def ws_infer(websocket: WebSocket) -> None:
    if websocket.query_params.get("token") != API_TOKEN:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    try:
        while True:
            datos = await websocket.receive_bytes()
            try:
                res = await run_in_threadpool(get_pipeline().inferir, datos, False, False)
            except Exception as exc:
                res = {"error": str(exc)}
            await websocket.send_text(json.dumps(res, ensure_ascii=False))
    except WebSocketDisconnect:
        pass
