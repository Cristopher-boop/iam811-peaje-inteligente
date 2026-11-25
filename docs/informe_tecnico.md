# Informe Técnico — Sistema Inteligente de Control de Peaje (IAM-811)

**Asignatura:** IAM-811 Machine Learning y Aprendizaje Profundo  
**Estudiante:** `Cristopher Diego Rodriguez Ayala` (`cristopherrodriguez1510@gmail.com`)  
**Fecha:** `2026-11-25`

---

## 1. Resumen Ejecutivo y Arquitectura

El sistema implementa el control automatizado de peaje vehicular de punta a punta sin depender de recursos locales durante la demostración ni revisión:

- **Captura Móvil (Flutter):** Inicia sesión mediante Google Sign-In (`id_token`), opera en *Modo Caseta* enviando frames de video (2–3 fps o disparo manual) y en *Modo Consulta* enviando placas en texto o fotografía al agente conversacional.
- **Aplicación Web y API REST (FastAPI + PostgreSQL en la nube):** Publicada en subdominio HTTPS (`https://peaje-iam811-web.onrender.com`), implementa OAuth 2.0 exclusivo con Google, lista blanca de usuarios con roles (`admin` / `operador`, incluyendo `doc.alvaro.pinto.ma@unifranz.edu.bo` como `admin`), lógica de tarifas por categoría y número de ejes, generación automática de alertas, panel en tiempo real mediante **Server-Sent Events (SSE)** y agente conversacional con LLM vía API OpenAI-compatible (OpenRouter, con respaldo determinístico SQL).
- **Inferencia Unificada en la Nube (RunPod GPU Pod):** Expone un único endpoint `POST /predict` protegido por token (`INFERENCE_URL`) que ejecuta secuencialmente **M1 (YOLOv9-t ONNX + OCR híbrido)**, **M2 (YOLO11s-seg)** y **M3 (YOLO11s-pose)**.

---

## 2. Datasets y Aumentaciones (Roboflow)

### 2.1 Modelo M2 — Segmentación de Instancias de Vehículos
- **Proyecto Roboflow:** `cristopher-diego-rodriguez-ayala/peaje-vehiculos-seg-6laub` (**Versión 1** publicada).
- **Clases (5):** `moto`, `auto`, `minibus`, `camion`, `bus`.
- **Origen de datos:** Combinación curada de transporte público boliviano (`lab2e3/transporte-publico-bolivia` v3 y `jaqueline-paola-coronel-condori/transporte-publico` v1) con tráfico multimodal (`helmet-o8tke/vehicle-segmentation-pyt1u` v1, excluyendo rickshaws y clases ajenas al peaje), normalizada en formato COCO de segmentación por instancias (`docs/resumen_datasets_roboflow.json`).
- **Cantidad de imágenes:**
  - Base subida: `1,493` train / `543` valid (`2,036` imágenes; instancias base: `auto: 3,557`, `moto: 1,007`, `camion: 843`, `minibus: 529`, `bus: 524`).
  - Versión 1 exportada (YOLO11-seg tras aumentación 2x): `2,986` train (`9,452` polígonos) / `543` valid (`1,727` polígonos) = **`3,529` imágenes / `11,179` polígonos**.
- **Preprocesamiento y aumentaciones aplicadas en Roboflow:** `Auto-Orient`, `Resize Stretch 640×640`, `Horizontal Flip`, `Brightness ±15%`, `Blur 1.0px`, `Rotate ±10°`.

### 2.2 Modelo M3 — Keypoints de Ruedas (Conteo de Ejes)
- **Proyecto Roboflow:** `cristopher-diego-rodriguez-ayala/peaje-llantas-keypoints` (**Versión 1** publicada).
- **Esquema:** 1 clase (`llanta`) con 5 keypoints ordenados (`centro`, `contacto`, `superior`, `izquierda`, `derecha`), `kpt_shape: [5, 3]`, `flip_idx: [0, 1, 2, 4, 3]`.
- **Origen de datos:** Combinación de `datacluster-labs-agryi/vehicle-wheel-detection` v1 (motos, vehículos ligeros y camiones `six_wheeler` / `eight_wheeler`), `faisal-sduw7/vehicle-wheel` v9 y muestras de buses/camiones pre-etiquetadas con `YOLOE` (`yoloe-11s-seg.pt`).
- **Cantidad de imágenes:**
  - Base subida: `390` train / `146` valid (`536` imágenes base con `1,183` llantas anotadas de 5 keypoints).
  - Versión 1 exportada (YOLO Pose tras aumentación 2x): `778` train (`1,682` llantas) / `146` valid (`341` llantas) = **`924` imágenes / `2,023` llantas** (`20` columnas por instancia verificadas).
- **Preprocesamiento y aumentaciones aplicadas en Roboflow:** `Auto-Orient`, `Resize Stretch 640×640`, `Horizontal Flip`, `Brightness ±15%`.

---

## 3. Entrenamiento y Métricas

### 3.1 M1 — Detección de Placas + OCR
- **Detector:** `yolo-v9-t-384-license-plates-end2end.onnx` (reutilizado).
- **OCR:** Pipeline híbrido `fast-plate-ocr` (`cct_xs_v2_global.onnx`) + `EasyOCR` con filtrado de altura de caracteres y normalización boliviana (`^\d{3,4}[A-Z]{3}$`).
- **Métricas medidas localmente (`docs/resumen_metricas_m1.json`):**
  - *Evaluación en autopista real* (`49` imágenes de `vehicle-segmentation-pyt1u` con placa anotada >= 36 px):
    - Tasa de detección de placa: **`95.92%`** (`47/49`, confianza promedio `0.7537`)
    - Tasa de lectura no vacía (híbrido / `fast-plate-ocr`): **`91.84%`** vs `EasyOCR` solo: `18.37%` (confianza OCR promedio `0.6151`)
  - *Evaluación controlada formato Bolivia* (`10` escenas con franja `"BOLIVIA"` + `^\d{3,4}[A-Z]{3}$`):
    - Detección: **`100%`** (`10/10`) | Exact Match híbrido: **`90.0%`** (`9/10`) vs `EasyOCR` `80.0%` | **CER promedio:** **`0.0143`**

### 3.2 M2 — Segmentación de Instancias (`yolo11s-seg`)
- **Métricas (`resumen_metricas_m2.json` tras ejecutar `entrenamiento/M2_segmentacion_vehiculos.ipynb` en Colab T4):**
  - `box mAP@50`: `[PENDIENTE EJECUTAR EN COLAB T4]` | `box mAP@50-95`: `[PENDIENTE]`
  - `mask mAP@50`: `[PENDIENTE EJECUTAR EN COLAB T4]` | `mask mAP@50-95`: `[PENDIENTE]`
  - `mask mAP@50` por clase (`moto`, `auto`, `minibus`, `camion`, `bus`): `[PENDIENTE]`

### 3.3 M3 — Keypoints de Ruedas y Conteo de Ejes (`yolo11s-pose`)
- **Métricas (`resumen_metricas_m3.json` tras ejecutar `entrenamiento/M3_keypoints_ruedas.ipynb` en Colab T4):**
  - `box mAP@50`: `[PENDIENTE EJECUTAR EN COLAB T4]`
  - `pose mAP@50 (OKS)`: `[PENDIENTE EJECUTAR EN COLAB T4]` | `pose mAP@50-95`: `[PENDIENTE]`
  - Error en conteo de ejes (`MAE` / exactitud): `[PENDIENTE]`

---

## 4. Latencia Punta a Punta y Tiempo Real

- **Mecanismo de tiempo real:** **Server-Sent Events (SSE)** sobre `/api/v1/stream` (con respaldo WebSocket `/api/v1/ws/panel`), elegido por su compatibilidad nativa con proxies HTTPS de PaaS y reconexión automática en navegadores sin requerir servidor broker separado.
- **Desglose de latencia promedio observada:**
  - M1 Detección + OCR (benchmark CPU local 49 imgs): **`164.05 ms`**
  - Pipeline completo `/predict` en CPU local (imagen real de bus `640×480`, `yolo11s-seg` respaldo + M1+OCR): **`381.0 ms`** (`m1_placa: 29.8 ms`, `m1_ocr: 164.5 ms`, `m2_seg: 155.3 ms`)
  - Total inferencia en GPU RunPod: `[PENDIENTE MEDIR TRAS INICIAR POD GPU]`
  - Punta a punta (App móvil → Web HTTPS Render → RunPod GPU → Neon PostgreSQL → Panel SSE): `[PENDIENTE MEDIR EN PRUEBA CON POD GPU ACTIVO]`

---

## 5. Gestión de Entornos (Bonificación `conda`, `mamba`, `venv`)

Los entornos están fijados en `entrenamiento/environment.yml`, `inferencia/environment.yml` / `requirements.txt` y `web/requirements.txt`, validados mediante `docs/diagnostico_entorno.py` en Google Colab, RunPod y el servidor web en la nube.

---

## 6. Problemas Encontrados y Soluciones

1. **Separación de leyenda `"BOLIVIA"` en placas bolivianas:** `fast-plate-ocr` no incluye Bolivia en su entrenamiento global; se integró respaldo `EasyOCR` filtrando caracteres cuya altura sea `< 60%` de la altura máxima de texto en el recorte.
2. **Doble conteo en ejes con llantas duales (tándem) o del lado lejano:** Se aplicó el filtro geométrico de altura visible (`UMBRAL_CERCANA = 0.65`) y agrupación horizontal por diámetro de llanta (`SEPARACION_EJE = 0.6`).
3. **Persistencia de imágenes de evidencia en PaaS efímero:** En lugar de guardar archivos en disco local del contenedor web, las imágenes anotadas se almacenan comprimidas en PostgreSQL (`evidencias`) servidas bajo autenticación en `/api/v1/evidencias/{id}.jpg`.
4. **Cuentas de Google bloqueadas antes de llegar al backend:** Configurar `doc.alvaro.pinto.ma@unifranz.edu.bo` tanto en la tabla `usuarios` como en *Test Users* de Google Cloud Console (o publicar la app OAuth) y registrar el `SHA-1` del APK para `google_sign_in`.
