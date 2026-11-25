# Guía de Despliegue Rápido en la Nube y Guion del Video (8–12 min)

## Parte 1 — Orden exacto de despliegue en la nube

### 1. Datasets y Entrenamiento (Roboflow + Colab)
1. En **Roboflow**, crea dos proyectos:
   - Proyecto 1 (*Instance Segmentation*): clases exactas `moto`, `auto`, `minibus`, `camion`, `bus`.
   - Proyecto 2 (*Keypoint Detection*): clase `llanta` con 5 keypoints en este orden exacto: `centro`, `contacto`, `superior`, `izquierda`, `derecha`.
2. Si necesitas acelerar el etiquetado de tus fotos propias:
   - Sube tus fotos a `entrenamiento/M2_segmentacion_vehiculos.ipynb` (Parte A, `EJECUTAR_PREETIQUETADO = True`) y `entrenamiento/M3_keypoints_ruedas.ipynb` (Parte A, `EJECUTAR_AUTOETIQUETADO = True`).
   - Descarga los `.zip` pre-etiquetados, súbelos a Roboflow, corrige las cajas/máscaras/keypoints y genera una versión con aumentaciones.
3. Ejecuta la Parte B y C de ambos notebooks en **Google Colab (GPU T4)** para obtener:
   - `best_m2_seg.pt` + `resumen_metricas_m2.json`
   - `best_m3_pose.pt` + `resumen_metricas_m3.json`
4. Ejecuta `entrenamiento/M1_validacion_ocr.ipynb` con tus fotos de placas para obtener `resumen_metricas_m1.json`.

### 2. Base de Datos PostgreSQL (Neon o Supabase)
1. Crea un proyecto gratuito en **Neon.tech** (o **Supabase**).
2. Copia la cadena `DATABASE_URL` (`postgresql://...`).

### 3. API Key de Qwen (OpenRouter o Alibaba DashScope)
1. En **OpenRouter** (`https://openrouter.ai`) crea una API Key.
2. Usa `QWEN_BASE_URL=https://openrouter.ai/api/v1` y `QWEN_MODEL=qwen/qwen-2.5-72b-instruct` (o `qwen/qwen3-32b`).

### 4. Subir repositorio a GitHub
1. Crea un repositorio en GitHub (ej. `iam811-peaje-inteligente`).
2. Verifica `git status` (asegúrate de NO incluir `.env`, `.keystore` ni credenciales) y haz `git push`.

### 5. Desplegar RunPod GPU Pod (Inferencia Unificada)
1. En RunPod, lanza un **GPU Pod** (ej. PyTorch 2.4 + CUDA 12.x) exponiendo el **puerto HTTP `8000`**.
2. En el terminal del Pod:
   ```bash
   cd /workspace
   git clone <URL-DE-TU-REPO> Lab3
   cd /workspace/Lab3/inferencia
   # Sube best_m2_seg.pt y best_m3_pose.pt dentro de /workspace/Lab3/inferencia/models/
   export API_TOKEN="peaje-token-secreto-2026"
   chmod +x start_runpod.sh
   ./start_runpod.sh
   ```
3. Copia la URL pública: `https://<RUNPOD_POD_ID>-8000.proxy.runpod.net`.

### 6. Desplegar Web + API REST en Render (Subdominio HTTPS)
1. En **Render.com**, crea un *Web Service* (o usa *Blueprint* con `render.yaml`) conectado a tu repo GitHub.
2. Copia la URL asignada por Render (ej. `https://peaje-iam811-web.onrender.com`).

### 7. Configurar Google Cloud Console (OAuth 2.0 Web + Android)
1. Crea o abre tu proyecto en **Google Cloud Console** → *APIs y servicios* → *Pantalla de consentimiento de OAuth* (tipo External).
   - Agrega tu correo y **`doc.alvaro.pinto.ma@unifranz.edu.bo`** en **Test Users** (o publica la app).
2. Crea **Credencial 1 (Aplicación Web)**:
   - Orígenes autorizados: `https://peaje-iam811-web.onrender.com`
   - URI de redirección autorizada: `https://peaje-iam811-web.onrender.com/auth/google/callback`
   - Copia `GOOGLE_CLIENT_ID` y `GOOGLE_CLIENT_SECRET` a las variables de entorno en Render.
3. Crea **Credencial 2 (Android)**:
   - Nombre del paquete: `com.unifranz.peaje_movil`
   - Huella digital `SHA-1`: la obtenida tras compilar el APK de `/movil`.
   - Copia `GOOGLE_ANDROID_CLIENT_ID` a Render.
4. En Render, completa `INFERENCE_URL`, `INFERENCE_API_TOKEN`, `ADMIN_EMAILS=doc.alvaro.pinto.ma@unifranz.edu.bo,tu-correo@gmail.com` y `QWEN_API_KEY`.

### 8. Compilar APK final con URL y Web Client ID definitivos
```bash
cd movil
flutter build apk --release --dart-define=BACKEND_URL=https://peaje-iam811-web.onrender.com --dart-define=GOOGLE_SERVER_CLIENT_ID=<GOOGLE_CLIENT_ID_WEB>
```

---

## Parte 2 — Guion del Video Demostración (8–12 minutos)

1. **[0:00 – 1:00] Presentación y arquitectura:** Tu nombre, objetivo del sistema y diagrama mostrando App Móvil → Web HTTPS (Render + PostgreSQL) → RunPod GPU (M1+M2+M3) + Qwen Web.
2. **[1:00 – 2:30] Google OAuth 2.0 y lista blanca (Web y Móvil):**
   - Muestra el intento de ingreso con una cuenta de Google **no autorizada** (en web muestra *Acceso Denegado* 403; en la app muestra el rechazo).
   - Inicia sesión con tu cuenta autorizada como administrador en web y en la app Android.
3. **[2:30 – 5:00] App móvil Modo Caseta + Panel en Tiempo Real (pantalla dividida):**
   - Abre la web en `/` (verificando badge *En vivo (SSE)*) junto a la pantalla del celular.
   - Apunta la cámara real de la app a vehículos/imágenes de prueba (auto, moto/bus/camión).
   - Muestra en el móvil: placa leída, categoría, ejes, tarifa y estado; y simultáneamente en la web cómo aparece la fila nueva con su imagen de evidencia y suben los contadores **sin recargar la página**.
4. **[5:00 – 6:30] Alerta en tiempo real:**
   - Captura una placa **no registrada** (y si puedes, una sin saldo) y muestra la alerta apareciendo al instante en el panel y en `/alertas`, luego márcala como atendida.
5. **[6:30 – 8:30] Agente Conversacional (4 estados + texto e imagen):**
   - Prueba desde `/chat` y desde *Modo Consulta* en el móvil:
     1. *Registrada y con paso* (`2481-KLM`)
     2. *Registrada sin paso* (`1895ABC`)
     3. *Identificada no registrada* (la placa capturada sin registro)
     4. *No existe* + prueba de error OCR (`24B1KLM` → "¿Quisiste decir 2481KLM?")
   - Envía también una **foto de placa** (y muestra qué pasa con una foto sin placa legible).
6. **[8:30 – 9:45] Consola de RunPod en vivo:**
   - Muestra la pestaña `/estado` en la web y la consola real de RunPod ejecutando Uvicorn y recibiendo una petición `POST /predict` en vivo.
7. **[9:45 – 10:30] Métricas y conclusiones:**
   - Resume mAP50 de M2 y M3, precisión de M1+OCR, latencia promedio y muestra los archivos de entornos (`conda`/`mamba`/`venv`) + salida de `diagnostico_entorno.py`.
