#!/usr/bin/env python3
"""Script de diagnostico de entorno para los servicios del sistema de peaje.

Ejecutar en cada entorno (Colab, RunPod, servidor web) para verificar versiones de
Python, gestor de entorno (venv/conda/mamba), PyTorch, CUDA y librerias clave.
"""
import argparse
import importlib
import json
import os
import platform
import sys
from datetime import datetime, timezone


PAQUETES = [
    "torch",
    "torchvision",
    "ultralytics",
    "cv2",
    "numpy",
    "onnxruntime",
    "fast_plate_ocr",
    "easyocr",
    "fastapi",
    "uvicorn",
    "sqlalchemy",
    "psycopg",
    "httpx",
    "openai",
    "roboflow",
]


def detectar_gestor_entorno() -> dict:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if conda_env:
        tipo = "mamba/conda"
    elif virtual_env:
        tipo = "venv"
    elif "google.colab" in sys.modules or os.path.exists("/content"):
        tipo = "colab-system"
    else:
        tipo = "system-python"
    return {
        "tipo": tipo,
        "conda_default_env": conda_env,
        "conda_prefix": conda_prefix,
        "virtual_env": virtual_env,
        "executable": sys.executable,
    }


def version_paquete(nombre: str) -> str:
    try:
        mod = importlib.import_module(nombre)
        return str(getattr(mod, "__version__", "instalado (sin __version__)"))
    except Exception as exc:
        return f"no disponible ({type(exc).__name__})"


def recopilar_diagnostico(servicio: str) -> dict:
    info = {
        "servicio": servicio,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sistema": {
            "os": platform.platform(),
            "python_version": sys.version.split()[0],
            "arquitectura": platform.machine(),
        },
        "entorno": detectar_gestor_entorno(),
        "cuda": {
            "disponible": False,
            "cuda_build": None,
            "gpu_count": 0,
            "gpu_name": None,
        },
        "paquetes": {pkg: version_paquete(pkg) for pkg in PAQUETES},
    }

    try:
        import torch

        disponible = bool(torch.cuda.is_available())
        info["cuda"] = {
            "disponible": disponible,
            "cuda_build": torch.version.cuda,
            "gpu_count": int(torch.cuda.device_count()) if disponible else 0,
            "gpu_name": torch.cuda.get_device_name(0) if disponible else None,
        }
    except Exception as exc:
        info["cuda"]["error"] = str(exc)

    try:
        import onnxruntime as ort

        info["onnxruntime_providers"] = ort.get_available_providers()
    except Exception:
        info["onnxruntime_providers"] = []

    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostico de entorno Python/CUDA/IA")
    parser.add_argument("--servicio", default="general", help="Nombre del servicio (colab, runpod, web)")
    parser.add_argument("--out", default="", help="Ruta opcional para guardar el JSON de salida")
    args = parser.parse_args()

    resultado = recopilar_diagnostico(args.servicio)
    texto = json.dumps(resultado, indent=2, ensure_ascii=False)
    print(texto)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(texto + "\n")


if __name__ == "__main__":
    main()
