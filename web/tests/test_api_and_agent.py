"""Pruebas integrales de API REST v1, autenticacion Google, tarifas, alertas y agente."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

# Fuerza BD SQLite aislada para pruebas antes de importar app
TEST_DB = Path(__file__).parent / "test_peaje.db"
if TEST_DB.exists():
    TEST_DB.unlink()
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB.as_posix()}"
os.environ["APP_ENV"] = "test"
os.environ["JWT_SECRET"] = "test-jwt-secret-1234567890"
os.environ["SESSION_SECRET"] = "test-session-secret-1234567890"
os.environ["QWEN_API_KEY"] = ""  # Prueba la ruta deterministica sin red externa

WEB_DIR = Path(__file__).resolve().parents[1]
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))

from app.database import engine  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c
    engine.dispose()
    if TEST_DB.exists():
        TEST_DB.unlink()


def _login_as(client: TestClient, email: str, name: str = "Usuario Prueba") -> dict:
    with patch("app.routers.api_v1.verificar_google_id_token", new=AsyncMock(return_value={"email": email, "name": name})):
        r = client.post("/api/v1/auth/google", json={"id_token": "fake-google-token"})
    return {"status_code": r.status_code, "json": r.json()}


def test_rutas_exigen_autenticacion(client: TestClient):
    assert client.get("/api/v1/vehiculos").status_code == 401
    assert client.post("/api/v1/placas/consultar", json={"placa": "2481KLM"}).status_code == 401
    assert client.post("/api/v1/agente/chat", json={"mensaje": "2481KLM"}).status_code == 401


def test_lista_blanca_y_roles_google(client: TestClient):
    rechazado = _login_as(client, "intruso@gmail.com")
    assert rechazado["status_code"] == 403

    admin = _login_as(client, "doc.alvaro.pinto.ma@unifranz.edu.bo", "Alvaro Pinto")
    assert admin["status_code"] == 200
    token_admin = admin["json"]["access_token"]
    assert admin["json"]["usuario"]["rol"] == "admin"

    headers_admin = {"Authorization": f"Bearer {token_admin}"}
    vehiculos = client.get("/api/v1/vehiculos", headers=headers_admin).json()["items"]
    assert len(vehiculos) >= 20

    # Crear operador y verificar restriccion RBAC
    r_op = client.post(
        "/api/v1/usuarios",
        headers=headers_admin,
        json={"email_google": "operador.peaje@gmail.com", "nombre": "Operador 1", "rol": "operador"},
    )
    assert r_op.status_code == 201

    op_login = _login_as(client, "operador.peaje@gmail.com", "Operador 1")
    assert op_login["status_code"] == 200
    headers_op = {"Authorization": f"Bearer {op_login['json']['access_token']}"}
    assert client.post(
        "/api/v1/vehiculos",
        headers=headers_op,
        json={"placa": "9900AAA", "propietario": "Test", "categoria": "auto", "ejes": 2, "saldo": 10},
    ).status_code == 403


def test_cuatro_estados_y_sugerencias_ocr(client: TestClient):
    token = _login_as(client, "doc.alvaro.pinto.ma@unifranz.edu.bo")["json"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Registrada y con paso
    r1 = client.post("/api/v1/agente/chat", headers=headers, json={"mensaje": "Consulta 2481-KLM"}).json()
    assert r1["estado_sistema"] == "Registrada y con paso"

    # 2. Registrada sin paso
    r2 = client.post("/api/v1/agente/chat", headers=headers, json={"mensaje": "1895ABC"}).json()
    assert r2["estado_sistema"] == "Registrada sin paso"

    # 3. Identificada no registrada
    r3 = client.post("/api/v1/agente/chat", headers=headers, json={"mensaje": "8888-EVD"}).json()
    assert r3["estado_sistema"] == "Identificada no registrada"

    # 4. No existe + confusion OCR (B -> 8)
    r4 = client.post("/api/v1/agente/chat", headers=headers, json={"mensaje": "24B1KLM"}).json()
    assert r4["estado_sistema"] == "No existe"
    assert any(s["placa"] == "2481KLM" for s in r4["consulta"]["sugerencias"])
    assert "Quisiste decir" in r4["respuesta"]


def test_registro_paso_frame_tarifa_y_alertas(client: TestClient):
    token = _login_as(client, "doc.alvaro.pinto.ma@unifranz.edu.bo")["json"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # A. Paso autorizado con cobro real
    fake_inf_ok = {
        "placa": {"texto": "2481KLM", "texto_crudo": "2481-KLM", "confianza": 0.93},
        "vehiculo": {"categoria": "auto", "confianza": 0.91},
        "ejes": 2,
        "latencia_ms": {"total": 75.0},
    }
    with patch("app.routers.api_v1.llamar_inferencia", new=AsyncMock(return_value=fake_inf_ok)):
        r_ok = client.post(
            "/api/v1/pasos/frame",
            headers=headers,
            data={"caseta_id": "1"},
            files={"file": ("frame.jpg", b"fake-jpeg-bytes", "image/jpeg")},
        )
    assert r_ok.status_code == 200
    body_ok = r_ok.json()
    assert body_ok["paso"]["estado"] == "AUTORIZADO"
    assert body_ok["paso"]["monto"] == 5.0
    assert body_ok["saldo_restante"] == 115.0

    # Verifica imagen de evidencia persistida en BD
    ev_url = body_ok["paso"]["imagen_url"]
    assert client.get(ev_url, headers=headers).status_code == 200

    # B. Paso placa no registrada -> genera alerta
    fake_inf_evasion = {
        "placa": {"texto": "9999XYZ", "texto_crudo": "9999XYZ", "confianza": 0.88},
        "vehiculo": {"categoria": "camion", "confianza": 0.90},
        "ejes": 3,
        "latencia_ms": {"total": 88.0},
    }
    with patch("app.routers.api_v1.llamar_inferencia", new=AsyncMock(return_value=fake_inf_evasion)):
        r_ev = client.post(
            "/api/v1/pasos/frame",
            headers=headers,
            data={"caseta_id": "1"},
            files={"file": ("frame.jpg", b"fake-jpeg-bytes", "image/jpeg")},
        )
    assert r_ev.status_code == 200
    assert r_ev.json()["paso"]["estado"] == "NO_REGISTRADA"
    assert r_ev.json()["paso"]["monto"] == 25.0
    assert r_ev.json()["alerta"]["tipo"] == "PLACA_NO_REGISTRADA"

    # C. Consultar imagen sin placa legible -> pide otra foto
    fake_inf_ilegible = {"placa": {"texto": "", "confianza": 0.0}}
    with patch("app.routers.api_v1.llamar_inferencia", new=AsyncMock(return_value=fake_inf_ilegible)):
        r_il = client.post(
            "/api/v1/placas/consultar-imagen",
            headers=headers,
            files={"file": ("borrosa.jpg", b"fake-jpeg-bytes", "image/jpeg")},
        )
    assert r_il.status_code == 200
    assert r_il.json()["placa_detectada"] is False

    # D. Resumen caseta e historial
    assert client.get("/api/v1/casetas/1/resumen", headers=headers).status_code == 200
    hist = client.get("/api/v1/placas/2481-KLM/historial", headers=headers).json()
    assert hist["total_pasos"] >= 2

    # E. Modo stream: frame sin vehiculo ni placa no ensucia BD, y cooldown evita doble cobro
    with patch("app.routers.api_v1.llamar_inferencia", new=AsyncMock(return_value={"placa": {"texto": ""}, "vehiculo": {}})):
        r_vacio = client.post(
            "/api/v1/pasos/frame",
            headers=headers,
            data={"caseta_id": "1", "modo": "stream"},
            files={"file": ("vacio.jpg", b"fake", "image/jpeg")},
        )
    assert r_vacio.status_code == 200
    assert r_vacio.json()["registrado"] is False
    assert r_vacio.json()["motivo"] == "sin_vehiculo"

    with patch("app.routers.api_v1.llamar_inferencia", new=AsyncMock(return_value=fake_inf_ok)):
        r_dup = client.post(
            "/api/v1/pasos/frame",
            headers=headers,
            data={"caseta_id": "1", "modo": "stream"},
            files={"file": ("dup.jpg", b"fake", "image/jpeg")},
        )
    assert r_dup.status_code == 200
    assert r_dup.json()["registrado"] is False
    assert r_dup.json()["motivo"] == "cooldown_placa"
    assert r_dup.json()["saldo_restante"] == 115.0
