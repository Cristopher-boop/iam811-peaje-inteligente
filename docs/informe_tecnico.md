# Informe Técnico — Sistema Inteligente de Control de Peaje (IAM-811)

**Asignatura:** IAM-811 Machine Learning y Aprendizaje Profundo  
**Estudiante:** `Cristopher Diego Rodriguez Ayala` (`cristopherrodriguez1510@gmail.com`)  
**Fecha:** `[COMPLETAR FECHA]`

---

## 1. Resumen Ejecutivo y Arquitectura

El sistema implementa el control automatizado de peaje vehicular de punta a punta sin depender de recursos locales durante la demostración ni revisión:

- **Captura Móvil (Flutter):** Inicia sesión mediante Google Sign-In (`id_token`), opera en *Modo Caseta* enviando frames de video (2–3 fps o disparo manual) y en *Modo Consulta* enviando placas en texto o fotografía al agente conversacional.
- **Aplicación Web y API REST (FastAPI + PostgreSQL en la nube):** Publicada en subdominio HTTPS (`[COMPLETAR URL]`), implementa OAuth 2.0 exclusivo con Google, lista blanca de usuarios con roles (`admin` / `operador`), lógica de tarifas por categoría y número de ejes, generación automática de alertas, panel en tiempo real mediante **Server-Sent Events (SSE)** y agente conversacional con **Qwen**.
- **Inferencia Unificada en la Nube (RunPod GPU Pod):** Expone un único endpoint `POST /predict` protegido por token (`INFERENCE_URL`) que ejecuta secuencialmente **M1 (YOLOv9-t ONNX + OCR híbrido)**, **M2 (YOLO11s-seg)** y **M3 (YOLO11s-pose)**.

---

## 2. Datasets y Aumentaciones (Roboflow)

### 2.1 Modelo M2 — Segmentación de Instancias de Vehículos
- **Clases (5):** `moto`, `auto`, `minibus`, `camion`, `bus`.
- **Origen de datos:** Combinación de datasets públicos de Roboflow Universe e imágenes propias del parque automotor local (autos, motocicletas, minibuses/trufis, buses y camiones), revisadas manualmente en Roboflow (*Instance Segmentation*).
- **Cantidad de imágenes:** `[COMPLETAR: ej. XXX train / XXX valid / XXX test]`.
- **Aumentaciones aplicadas en Roboflow:** `[COMPLETAR: ej. Horizontal Flip, Brightness ±15%, Blur hasta 1.0px, Rotación ±10°]`.

### 2.2 Modelo M3 — Keypoints de Ruedas (Conteo de Ejes)
- **Esquema:** 1 clase (`llanta`) con 5 keypoints ordenados (`centro`, `contacto`, `superior`, `izquierda`, `derecha`), `kpt_shape: [5, 3]`, `flip_idx: [0, 1, 2, 4, 3]`.
- **Origen de datos:** >=200 imágenes laterales y en perspectiva 3/4 (autos, motos, minibuses, buses y camiones de 2, 3 y 4+ ejes), pre-etiquetadas en Colab y corregidas manualmente en Roboflow (*Keypoint Detection*).
- **Cantidad de imágenes:** `[COMPLETAR]`.
- **Aumentaciones aplicadas:** `[COMPLETAR]`.

---

## 3. Entrenamiento y Métricas

### 3.1 M1 — Detección de Placas + OCR
- **Detector:** `yolo-v9-t-384-license-plates-end2end.onnx` (reutilizado).
- **OCR:** Pipeline híbrido `fast-plate-ocr` (`cct_xs_v2_global.onnx`) + `EasyOCR` con filtrado de altura de caracteres y normalización boliviana (`^\d{3,4}[A-Z]{3}$`).
- **Métricas medidas (`resumen_metricas_m1.json`):**
  - Tasa de detección de placa: `[COMPLETAR %]`
  - Exactitud OCR (Exact Match): `[COMPLETAR %]`
  - Character Error Rate (CER): `[COMPLETAR]`

### 3.2 M2 — Segmentación de Instancias (`yolo11s-seg`)
- **Métricas (`resumen_metricas_m2.json`):**
  - `box mAP@50`: `[COMPLETAR]` | `box mAP@50-95`: `[COMPLETAR]`
  - `mask mAP@50`: `[COMPLETAR]` | `mask mAP@50-95`: `[COMPLETAR]`
  - `mask mAP@50` por clase (`moto`, `auto`, `minibus`, `camion`, `bus`): `[COMPLETAR]`

### 3.3 M3 — Keypoints de Ruedas y Conteo de Ejes (`yolo11s-pose`)
- **Métricas (`resumen_metricas_m3.json`):**
  - `box mAP@50`: `[COMPLETAR]`
  - `pose mAP@50 (OKS)`: `[COMPLETAR]` | `pose mAP@50-95`: `[COMPLETAR]`
  - Error en conteo de ejes (`MAE` / exactitud): `[COMPLETAR]`

---

## 4. Latencia Punta a Punta y Tiempo Real

- **Mecanismo de tiempo real:** **Server-Sent Events (SSE)** sobre `/api/v1/stream` (con respaldo WebSocket `/api/v1/ws/panel`), elegido por su compatibilidad nativa con proxies HTTPS de PaaS y reconexión automática en navegadores sin requerir servidor broker separado.
- **Desglose de latencia promedio observada:**
  - M1 Detección + OCR: `[COMPLETAR ms]`
  - M2 Segmentación: `[COMPLETAR ms]`
  - M3 Keypoints + conteo de ejes: `[COMPLETAR ms]`
  - Total inferencia en GPU RunPod: `[COMPLETAR ms]`
  - Punta a punta (App móvil → Web HTTPS → RunPod → BD PostgreSQL → Panel SSE): `[COMPLETAR ms]`

---

## 5. Gestión de Entornos (Bonificación `conda`, `mamba`, `venv`)

Los entornos están fijados en `entrenamiento/environment.yml`, `inferencia/environment.yml` / `requirements.txt` y `web/requirements.txt`, validados mediante `docs/diagnostico_entorno.py` en Google Colab, RunPod y el servidor web en la nube.

---

## 6. Problemas Encontrados y Soluciones

1. **Separación de leyenda `"BOLIVIA"` en placas bolivianas:** `fast-plate-ocr` no incluye Bolivia en su entrenamiento global; se integró respaldo `EasyOCR` filtrando caracteres cuya altura sea `< 60%` de la altura máxima de texto en el recorte.
2. **Doble conteo en ejes con llantas duales (tándem) o del lado lejano:** Se aplicó el filtro geométrico de altura visible (`UMBRAL_CERCANA = 0.65`) y agrupación horizontal por diámetro de llanta (`SEPARACION_EJE = 0.6`).
3. **Persistencia de imágenes de evidencia en PaaS efímero:** En lugar de guardar archivos en disco local del contenedor web, las imágenes anotadas se almacenan comprimidas en PostgreSQL (`evidencias`) servidas bajo autenticación en `/api/v1/evidencias/{id}.jpg`.
4. **Cuentas de Google bloqueadas antes de llegar al backend:** Configurar `doc.alvaro.pinto.ma@unifranz.edu.bo` tanto en la tabla `usuarios` como en *Test Users* de Google Cloud Console (o publicar la app OAuth) y registrar el `SHA-1` del APK para `google_sign_in`.
