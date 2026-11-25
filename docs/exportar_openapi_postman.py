"""Exporta el esquema OpenAPI (Swagger) y genera la coleccion Postman v2.1 del sistema."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
WEB_DIR = DOCS_DIR.parent / "web"

os.environ.setdefault("DATABASE_URL", f"sqlite:///{(DOCS_DIR / '_schema_tmp.db').as_posix()}")
os.environ.setdefault("APP_ENV", "development")

if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))

from app.main import app  # noqa: E402


def build_postman_collection(openapi_spec: dict) -> dict:
    items = []
    for path, methods in sorted(openapi_spec.get("paths", {}).items()):
        if not path.startswith("/api/v1"):
            continue
        for method, meta in methods.items():
            method_u = method.upper()
            is_public = path == "/api/v1/auth/google"
            raw_url = "{{base_url}}" + path.replace("{placa}", "2481KLM").replace("{caseta_id}", "1").replace("{alerta_id}", "1").replace("{vehiculo_id}", "1").replace("{usuario_id}", "1").replace("{evidencia_id}", "1")
            req: dict = {
                "method": method_u,
                "header": [],
                "url": {
                    "raw": raw_url,
                    "host": ["{{base_url}}"],
                    "path": [p for p in raw_url.replace("{{base_url}}/", "").split("/") if p],
                },
            }
            if not is_public:
                req["header"].append({"key": "Authorization", "value": "Bearer {{jwt_token}}", "type": "text"})

            if method_u in ("POST", "PUT", "PATCH") and "requestBody" in meta:
                content = meta["requestBody"].get("content", {})
                if "application/json" in content:
                    req["header"].append({"key": "Content-Type", "value": "application/json", "type": "text"})
                    sample = {}
                    if path == "/api/v1/auth/google":
                        sample = {"id_token": "<GOOGLE_ID_TOKEN>"}
                    elif path == "/api/v1/placas/consultar":
                        sample = {"placa": "2481-KLM"}
                    elif path == "/api/v1/agente/chat":
                        sample = {"mensaje": "Consulta el estado de la placa 2481-KLM"}
                    req["body"] = {"mode": "raw", "raw": json.dumps(sample, indent=2, ensure_ascii=False)}
                elif "multipart/form-data" in content:
                    req["body"] = {
                        "mode": "formdata",
                        "formdata": [
                            {"key": "file", "type": "file", "src": ""},
                            {"key": "caseta_id", "value": "1", "type": "text"},
                        ],
                    }

            items.append({
                "name": f"{method_u} {path} — {meta.get('summary', '')}".strip(" —"),
                "request": req,
            })

    return {
        "info": {
            "name": "Peaje Inteligente IAM-811 API v1",
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
            "description": openapi_spec.get("info", {}).get("description", ""),
        },
        "variable": [
            {"key": "base_url", "value": "https://peaje-iam811-web.onrender.com"},
            {"key": "jwt_token", "value": ""},
        ],
        "item": items,
    }


def main() -> None:
    spec = app.openapi()
    (DOCS_DIR / "openapi.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    postman = build_postman_collection(spec)
    (DOCS_DIR / "postman_collection.json").write_text(json.dumps(postman, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_db = DOCS_DIR / "_schema_tmp.db"
    if tmp_db.exists():
        tmp_db.unlink()
    print(f"Exportados docs/openapi.json y docs/postman_collection.json ({len(postman['item'])} endpoints)")


if __name__ == "__main__":
    main()
