#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

if [ -z "${API_TOKEN:-}" ]; then
  echo "[AVISO] API_TOKEN no definido; exporta API_TOKEN antes de iniciar para mantener un token fijo."
fi

# Crear/usar venv dedicado sobre Python del Pod (hereda torch/CUDA del sistema si usa --system-site-packages)
if [ ! -d ".venv" ]; then
  python3 -m venv --system-site-packages .venv
  source .venv/bin/activate
  pip install --no-cache-dir --upgrade pip
  pip install --no-cache-dir fastapi==0.165.0 "uvicorn[standard]==0.54.0" python-multipart==0.0.34 \
    ultralytics==8.4.218 onnxruntime-gpu==1.32.0 fast-plate-ocr==1.1.0 easyocr==1.7.2 \
    opencv-python-headless==5.1.0.95 pyyaml==6.0.4
else
  source .venv/bin/activate
fi

# Evidencia de diagnostico de entorno para bonificacion
if [ -f "../docs/diagnostico_entorno.py" ]; then
  python ../docs/diagnostico_entorno.py --servicio runpod_inferencia --out ../docs/diagnostico_runpod.json
fi

PORT="${PORT:-8000}"
if [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "INFERENCE_URL=https://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net"
fi

exec uvicorn app:app --host 0.0.0.0 --port "${PORT}"
