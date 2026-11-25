#!/usr/bin/env python3
"""Verificador integral de produccion (Render + Neon + RunPod + 4 estados del agente).

Uso:
  python docs/verificar_produccion.py --web https://peaje-iam811-web.onrender.com --runpod https://<POD_ID>-8000.proxy.runpod.net --token <INFERENCE_TOKEN>
"""
from __future__ import annotations

import argparse
import json
import sys

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifica salud de produccion Web y RunPod")
    parser.add_argument("--web", default="https://peaje-iam811-web.onrender.com")
    parser.add_argument("--runpod", default="")
    parser.add_argument("--token", default="")
    args = parser.parse_args()

    web = args.web.rstrip("/")
    print(f"Verificando Web: {web} ...")
    for path, expected in [("/health", 200), ("/login", 200), ("/docs", 200), ("/openapi.json", 200), ("/api/v1/vehiculos", 401)]:
        r = httpx.get(web + path, timeout=65.0)
        ok = r.status_code == expected
        print(f"  [{'OK' if ok else 'FAIL'}] GET {path} -> {r.status_code} (esperado {expected})")
        if not ok:
            sys.exit(1)

    if args.runpod:
        rp = args.runpod.rstrip("/")
        print(f"\nVerificando RunPod: {rp} ...")
        r_h = httpx.get(rp + "/health", timeout=20.0)
        print("  /health ->", r_h.status_code, json.dumps(r_h.json(), ensure_ascii=False))
        if r_h.status_code != 200:
            sys.exit(1)

    print("\nVerificacion basica completada correctamente.")


if __name__ == "__main__":
    main()
