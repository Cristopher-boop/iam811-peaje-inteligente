"""Pipeline unificado de inferencia para control de peaje (M1 + OCR, M2, M3).

Combina:
- M1: Deteccion de placas (YOLOv9-t ONNX) + OCR hibrido (fast-plate-ocr + EasyOCR).
- M2: Segmentacion de instancias de vehiculos (YOLO11-seg: moto, auto, minibus, camion, bus).
- M3: Deteccion de keypoints de ruedas (YOLO11-pose, 5 keypoints/rueda) y conteo de ejes.
"""
from __future__ import annotations

import base64
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch

logger = logging.getLogger("peaje.inferencia")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("MODELS_DIR", BASE_DIR / "models"))

LADO_PLACA = 384
FORMATO_BOLIVIA = re.compile(r"^\d{3,4}[A-Z]{3}$")
PERMITIDOS_OCR = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

CLASES_M2 = ("moto", "auto", "minibus", "camion", "bus")
NORMALIZAR_CLASE = {
    "moto": "moto",
    "motorcycle": "moto",
    "motorbike": "moto",
    "auto": "auto",
    "car": "auto",
    "sedan": "auto",
    "suv": "auto",
    "minibus": "minibus",
    "minibús": "minibus",
    "van": "minibus",
    "camion": "camion",
    "camión": "camion",
    "truck": "camion",
    "bus": "bus",
}
COLORES_BGR = {
    "moto": (156, 188, 26),
    "auto": (219, 152, 52),
    "minibus": (113, 204, 46),
    "camion": (18, 156, 243),
    "bus": (60, 76, 231),
}

# Constantes geometricas de keypoints de ruedas (keypoints_llantas_yolo_detectron2_v3.ipynb)
KPT_COLORES_BGR = [
    (255, 255, 255),  # 0: centro
    (0, 0, 255),      # 1: contacto
    (255, 200, 0),    # 2: superior
    (0, 255, 0),      # 3: izquierda
    (255, 0, 255),    # 4: derecha
]
ESQUELETO_LLANTA = [(0, 1), (0, 2), (0, 3), (0, 4)]
UMBRAL_CONTENCION = 0.7
UMBRAL_CERCANA = 0.65
SEPARACION_EJE = 0.6
UMBRAL_KP = 0.05


def limpiar_placa(texto: str) -> str:
    """Normaliza el texto de placa a mayusculas alfanumericas."""
    return re.sub(r"[^A-Z0-9]", "", (texto or "").upper())


def _altura_visible(kps: np.ndarray) -> float:
    return max(1.0, float(kps[1, 1] - kps[2, 1]))


def _contenida(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area


def _limpiar_llantas(llantas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    orden = sorted(llantas, key=lambda l: -(l["box"][2] - l["box"][0]) * (l["box"][3] - l["box"][1]))
    res: list[dict[str, Any]] = []
    for ll in orden:
        if all(_contenida(ll["box"], r["box"]) < UMBRAL_CONTENCION for r in res):
            res.append(ll)
    return res


def _analizar_ejes(vehiculo: dict[str, Any], llantas_v: list[dict[str, Any]]) -> tuple[int | None, str, str]:
    if not llantas_v:
        if vehiculo.get("clase") == "moto":
            return 2, "estimado", "Moto sin ruedas visibles (2 ejes por categoria)"
        return None, "-", "Sin llantas detectadas"

    hmax = max(_altura_visible(l["kps"]) for l in llantas_v)
    cercanas = [l for l in llantas_v if _altura_visible(l["kps"]) >= UMBRAL_CERCANA * hmax]

    x1, y1, x2, y2 = vehiculo["box"]
    proporcion = (x2 - x1) / max(1.0, y2 - y1)
    if len(cercanas) == 2 and proporcion < 1.6 and vehiculo.get("clase") != "moto":
        a, b = cercanas
        angostas = all((l["kps"][4, 0] - l["kps"][3, 0]) / _altura_visible(l["kps"]) < 0.65 for l in cercanas)
        misma_altura = abs(a["kps"][1, 1] - b["kps"][1, 1]) < 0.15 * hmax
        if angostas and misma_altura:
            return None, "frontal/posterior", "No determinable en esta vista"

    xs = sorted(float(l["kps"][0, 0]) for l in cercanas)
    diam = float(np.median([_altura_visible(l["kps"]) for l in cercanas]))
    ejes = 1 + sum(1 for p, q in zip(xs, xs[1:]) if q - p > SEPARACION_EJE * diam)
    redondez = float(np.median([(l["kps"][4, 0] - l["kps"][3, 0]) / _altura_visible(l["kps"]) for l in cercanas]))
    vista = "lateral" if redondez > 0.8 else "3/4"
    nota = ""
    if vehiculo.get("clase") == "moto":
        ejes = 2
    elif ejes < 2:
        ejes, nota = 2, "Minimo asumido (solo se detecto 1 eje)"
    return int(ejes), vista, nota


class TollInferencePipeline:
    """Pipeline unificado con bloqueo por inferencia para ejecucion segura sobre GPU."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.m1_conf = float(os.environ.get("M1_CONF", "0.35"))
        self.m2_conf = float(os.environ.get("M2_CONF", "0.30"))
        self.m3_conf = float(os.environ.get("M3_CONF", "0.25"))

        self.sesion_m1: ort.InferenceSession | None = None
        self.lector_fast = None
        self.lector_easy = None
        self.modelo_m2 = None
        self.modelo_m3 = None
        self.m2_is_trained = False
        self.m3_is_trained = False

        self._cargar_modelos()

    def _cargar_modelos(self) -> None:
        # 1. M1 Deteccion (ONNX)
        det_path = MODELS_DIR / "yolo-v9-t-384-license-plates-end2end.onnx"
        if not det_path.exists():
            raise FileNotFoundError(f"No se encontro el detector M1 en {det_path}")
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self.sesion_m1 = ort.InferenceSession(str(det_path), providers=providers)
        logger.info("M1 detector cargado (%s)", providers)

        # 2. M1 OCR (fast-plate-ocr + EasyOCR)
        ocr_onnx = MODELS_DIR / "cct_xs_v2_global.onnx"
        ocr_yaml = MODELS_DIR / "cct_xs_v2_global_plate_config.yaml"
        if ocr_onnx.exists() and ocr_yaml.exists():
            try:
                from fast_plate_ocr import LicensePlateRecognizer

                self.lector_fast = LicensePlateRecognizer(
                    onnx_model_path=str(ocr_onnx),
                    plate_config_path=str(ocr_yaml),
                )
                logger.info("fast-plate-ocr cargado")
            except Exception as exc:
                logger.warning("No se pudo cargar fast-plate-ocr: %s", exc)

        try:
            import easyocr

            easy_dir = MODELS_DIR / "easyocr"
            easy_dir.mkdir(parents=True, exist_ok=True)
            self.lector_easy = easyocr.Reader(
                ["en"],
                gpu=self.device == "cuda",
                verbose=False,
                model_storage_directory=str(easy_dir),
            )
            logger.info("EasyOCR cargado (gpu=%s)", self.device == "cuda")
        except Exception as exc:
            logger.warning("No se pudo cargar EasyOCR: %s", exc)

        # 3. M2 Segmentacion de vehiculos + M3 Keypoints de ruedas
        from ultralytics import YOLO

        m2_path = Path(os.environ.get("M2_WEIGHTS", str(MODELS_DIR / "best_m2_seg.pt")))
        if not m2_path.is_absolute():
            m2_path = BASE_DIR / m2_path
        if m2_path.exists():
            self.modelo_m2 = YOLO(str(m2_path))
            self.m2_is_trained = True
            logger.info("M2 propio cargado: %s", m2_path)
        else:
            logger.warning("Peso M2 propio no encontrado en %s; usando respaldo yolo11s-seg.pt", m2_path)
            self.modelo_m2 = YOLO("yolo11s-seg.pt")

        m3_path = Path(os.environ.get("M3_WEIGHTS", str(MODELS_DIR / "best_m3_pose.pt")))
        if not m3_path.is_absolute():
            m3_path = BASE_DIR / m3_path
        if m3_path.exists():
            self.modelo_m3 = YOLO(str(m3_path))
            self.m3_is_trained = True
            logger.info("M3 propio cargado: %s", m3_path)
        else:
            logger.warning("Peso M3 propio no encontrado en %s; M3 en modo degradado hasta subir best_m3_pose.pt", m3_path)
            self.modelo_m3 = None

        self._calentar()

    def _calentar(self) -> None:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            self.detectar_placas(dummy)
            if self.modelo_m2 is not None:
                self.modelo_m2.predict(dummy, conf=self.m2_conf, device=self.device, verbose=False)
            if self.modelo_m3 is not None:
                self.modelo_m3.predict(dummy, conf=self.m3_conf, device=self.device, verbose=False)
        except Exception as exc:
            logger.warning("Error durante warmup: %s", exc)

    def estado_salud(self) -> dict[str, Any]:
        return {
            "estado": "ok",
            "dispositivo": self.device,
            "cuda_disponible": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
            "modelos": {
                "m1_deteccion": self.sesion_m1 is not None,
                "m1_fast_ocr": self.lector_fast is not None,
                "m1_easyocr": self.lector_easy is not None,
                "m2_segmentacion": self.modelo_m2 is not None,
                "m2_entrenado_propio": self.m2_is_trained,
                "m3_keypoints": self.modelo_m3 is not None,
                "m3_entrenado_propio": self.m3_is_trained,
            },
        }

    def detectar_placas(self, imagen_bgr: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        alto, ancho = imagen_bgr.shape[:2]
        escala = min(LADO_PLACA / alto, LADO_PLACA / ancho)
        nw, nh = round(ancho * escala), round(alto * escala)
        mx, my = (LADO_PLACA - nw) / 2, (LADO_PLACA - nh) / 2
        lienzo = cv2.resize(imagen_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        lienzo = cv2.copyMakeBorder(
            lienzo,
            round(my - 0.1),
            round(my + 0.1),
            round(mx - 0.1),
            round(mx + 0.1),
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        entrada = (lienzo[:, :, ::-1].transpose(2, 0, 1) / 255.0).astype(np.float32)[None]
        assert self.sesion_m1 is not None
        salida = self.sesion_m1.run(None, {self.sesion_m1.get_inputs()[0].name: entrada})[0]
        cajas = []
        for _, x1, y1, x2, y2, _, conf in salida:
            if float(conf) < self.m1_conf:
                continue
            rx1, rx2 = [(v - mx) / escala for v in (x1, x2)]
            ry1, ry2 = [(v - my) / escala for v in (y1, y2)]
            cajas.append((
                int(max(0, rx1)),
                int(max(0, ry1)),
                int(min(ancho, rx2)),
                int(min(alto, ry2)),
                float(conf),
            ))
        cajas.sort(key=lambda c: -c[4])
        return cajas

    def leer_ocr(self, recorte_bgr: np.ndarray) -> tuple[str, str, float, str]:
        if recorte_bgr.size == 0:
            return "", "", 0.0, "ninguno"

        candidatos: list[tuple[str, str, float, str]] = []
        if self.lector_fast is not None:
            rgb = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2RGB)
            pred = self.lector_fast.run_one(rgb, return_confidence=True)
            raw = pred.plate or ""
            norm = limpiar_placa(raw)
            conf = float(pred.char_probs.mean()) if pred.char_probs is not None else 0.5
            candidatos.append((norm, raw, conf, "fast-plate-ocr"))

        if self.lector_easy is not None:
            lecturas = self.lector_easy.readtext(recorte_bgr, allowlist=PERMITIDOS_OCR)
            if lecturas:
                altura = lambda l: max(p[1] for p in l[0]) - min(p[1] for p in l[0])
                hmax = max(altura(l) for l in lecturas)
                lecturas = [l for l in lecturas if altura(l) >= hmax * 0.6]
                lecturas.sort(key=lambda l: l[0][0][0])
                raw_e = "".join(t for _, t, _ in lecturas)
                norm_e = limpiar_placa(raw_e)
                conf_e = float(sum(c for _, _, c in lecturas) / len(lecturas))
                candidatos.append((norm_e, raw_e, conf_e, "easyocr"))

        if not candidatos:
            return "", "", 0.0, "ninguno"

        # Prioriza coincidencias validas con formato boliviano y mayor longitud/confianza
        candidatos.sort(
            key=lambda c: (
                bool(FORMATO_BOLIVIA.match(c[0])),
                len(c[0]) >= 5,
                c[2],
            ),
            reverse=True,
        )
        return candidatos[0]

    def _segmentar_vehiculos(self, img_bgr: np.ndarray) -> list[dict[str, Any]]:
        if self.modelo_m2 is None:
            return []
        r = self.modelo_m2.predict(img_bgr, conf=self.m2_conf, device=self.device, agnostic_nms=True, verbose=False)[0]
        if r.masks is None or len(r.boxes) == 0:
            return []
        H, W = r.orig_shape
        area_total = float(max(1, H * W))
        clases = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        cajas = r.boxes.xyxy.cpu().numpy()
        vehiculos = []
        for poly, cid, s, b in zip(r.masks.xy, clases, confs, cajas):
            nombre_raw = str(r.names[cid]).strip().lower()
            clase = NORMALIZAR_CLASE.get(nombre_raw)
            if clase is None or len(poly) < 3:
                continue
            mask = np.zeros((H, W), np.uint8)
            cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
            vehiculos.append({
                "clase": clase,
                "conf": float(s),
                "box": [float(v) for v in b],
                "mask": mask.astype(bool),
                "area_mascara": round(float(mask.sum()) / area_total, 4),
            })
        vehiculos.sort(key=lambda v: -v["area_mascara"] * v["conf"])
        return vehiculos

    def _detectar_llantas(self, img_bgr: np.ndarray) -> list[dict[str, Any]]:
        if self.modelo_m3 is None:
            return []
        r = self.modelo_m3.predict(img_bgr, conf=self.m3_conf, device=self.device, verbose=False)[0]
        llantas = []
        if r.keypoints is not None and len(r.boxes):
            xy = r.keypoints.xy.cpu().numpy()
            kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(xy.shape[:2])
            for b, s, p, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), xy, kc):
                if p.shape[0] < 5:
                    continue
                llantas.append({
                    "conf": float(s),
                    "box": [float(v) for v in b],
                    "kps": np.concatenate([p[:5], c[:5, None]], axis=1),
                })
        return _limpiar_llantas(llantas)

    @staticmethod
    def _seleccionar_vehiculo_principal(
        vehiculos: list[dict[str, Any]],
        placa_bbox: list[int] | None,
    ) -> dict[str, Any] | None:
        if not vehiculos:
            return None
        if placa_bbox:
            px = (placa_bbox[0] + placa_bbox[2]) / 2.0
            py = (placa_bbox[1] + placa_bbox[3]) / 2.0
            contienen = [
                v for v in vehiculos
                if v["box"][0] <= px <= v["box"][2] and v["box"][1] <= py <= v["box"][3]
            ]
            if contienen:
                return max(contienen, key=lambda v: v["area_mascara"])
        return vehiculos[0]

    @staticmethod
    def _asociar_llantas(
        vehiculo: dict[str, Any] | None,
        llantas: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not vehiculo:
            return llantas
        x1, y1, x2, y2 = vehiculo["box"]
        w, h = x2 - x1, y2 - y1
        asociadas = []
        for ll in llantas:
            cx, cy = ll["kps"][0, :2]
            if x1 - 0.05 * w <= cx <= x2 + 0.05 * w and y1 + 0.30 * h <= cy <= y2 + 0.12 * h:
                asociadas.append(ll)
        return asociadas

    @staticmethod
    def _dibujar_evidencia(
        img_bgr: np.ndarray,
        placa_info: dict[str, Any] | None,
        vehiculo: dict[str, Any] | None,
        llantas: list[dict[str, Any]],
        ejes: int | None,
    ) -> str:
        vis = img_bgr.copy()
        g = max(2, int(round(max(vis.shape[:2]) / 450)))

        if vehiculo is not None:
            col = COLORES_BGR.get(vehiculo["clase"], (255, 200, 0))
            overlay = vis.copy()
            overlay[vehiculo["mask"]] = col
            vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
            cnts, _ = cv2.findContours(vehiculo["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, col, g)
            vx1, vy1, vx2, vy2 = map(int, vehiculo["box"])
            cv2.rectangle(vis, (vx1, vy1), (vx2, vy2), col, g)
            ejes_txt = f"{ejes} ejes" if ejes is not None else "ejes N/D"
            lbl = f"{vehiculo['clase'].upper()} {vehiculo['conf']:.2f} | {ejes_txt}"
            cv2.rectangle(vis, (vx1, max(0, vy1 - 26)), (vx1 + len(lbl) * 11, vy1), col, -1)
            cv2.putText(vis, lbl, (vx1 + 4, max(18, vy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        for ll in llantas:
            k = ll["kps"]
            lx1, ly1, lx2, ly2 = map(int, ll["box"])
            cv2.rectangle(vis, (lx1, ly1), (lx2, ly2), (0, 255, 255), max(1, g // 2))
            for a, b in ESQUELETO_LLANTA:
                if k[a, 2] > UMBRAL_KP and k[b, 2] > UMBRAL_KP:
                    cv2.line(vis, tuple(map(int, k[a, :2])), tuple(map(int, k[b, :2])), (0, 255, 255), max(1, g // 2), cv2.LINE_AA)
            for idx_k in range(len(k)):
                if k[idx_k, 2] > UMBRAL_KP:
                    cv2.circle(vis, tuple(map(int, k[idx_k, :2])), g + 2, KPT_COLORES_BGR[idx_k], -1, cv2.LINE_AA)
                    cv2.circle(vis, tuple(map(int, k[idx_k, :2])), g + 2, (0, 0, 0), 1, cv2.LINE_AA)

        if placa_info and placa_info.get("bbox"):
            px1, py1, px2, py2 = placa_info["bbox"]
            cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 220, 0), g + 1)
            txt = placa_info.get("texto") or "PLACA"
            cv2.putText(vis, txt, (px1, max(24, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 0), 2, cv2.LINE_AA)

        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return ""
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

    def inferir(
        self,
        datos_imagen: bytes,
        solo_placa: bool = False,
        incluir_imagen: bool = True,
    ) -> dict[str, Any]:
        arr = np.frombuffer(datos_imagen, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("No se pudo decodificar la imagen enviada.")

        H, W = img.shape[:2]
        t_total_0 = time.perf_counter()

        with self._lock:
            # --- M1: Deteccion de placa + OCR ---
            t_m1_0 = time.perf_counter()
            cajas_placa = self.detectar_placas(img)
            t_m1_det = (time.perf_counter() - t_m1_0) * 1000

            placa_data: dict[str, Any] = {
                "texto": "",
                "texto_crudo": "",
                "confianza": 0.0,
                "confianza_deteccion": 0.0,
                "formato_bolivia": False,
                "bbox": None,
                "lector": "ninguno",
            }
            t_ocr = 0.0
            if cajas_placa:
                px1, py1, px2, py2, pconf = cajas_placa[0]
                pad_x = max(2, int((px2 - px1) * 0.05))
                pad_y = max(2, int((py2 - py1) * 0.05))
                cx1, cy1 = max(0, px1 - pad_x), max(0, py1 - pad_y)
                cx2, cy2 = min(W, px2 + pad_x), min(H, py2 + pad_y)

                t_ocr_0 = time.perf_counter()
                txt_norm, txt_raw, ocr_conf, lector_usado = self.leer_ocr(img[cy1:cy2, cx1:cx2])
                t_ocr = (time.perf_counter() - t_ocr_0) * 1000

                placa_data = {
                    "texto": txt_norm,
                    "texto_crudo": txt_raw,
                    "confianza": round(ocr_conf, 3),
                    "confianza_deteccion": round(pconf, 3),
                    "formato_bolivia": bool(FORMATO_BOLIVIA.match(txt_norm)),
                    "bbox": [px1, py1, px2, py2],
                    "lector": lector_usado,
                }

            if solo_placa:
                total_ms = round((time.perf_counter() - t_total_0) * 1000, 1)
                return {
                    "placa": placa_data,
                    "latencia_ms": {"total": total_ms, "m1_placa": round(t_m1_det, 1), "m1_ocr": round(t_ocr, 1)},
                    "imagen_anotada": self._dibujar_evidencia(img, placa_data, None, [], None) if incluir_imagen else None,
                }

            # --- M2: Segmentacion de instancias ---
            t_m2_0 = time.perf_counter()
            vehiculos = self._segmentar_vehiculos(img)
            t_m2 = (time.perf_counter() - t_m2_0) * 1000
            vehiculo_principal = self._seleccionar_vehiculo_principal(vehiculos, placa_data["bbox"])

            # --- M3: Keypoints de ruedas y conteo de ejes ---
            t_m3_0 = time.perf_counter()
            todas_llantas = self._detectar_llantas(img)
            llantas_v = self._asociar_llantas(vehiculo_principal, todas_llantas)
            if vehiculo_principal is not None:
                ejes, vista, nota_ejes = _analizar_ejes(vehiculo_principal, llantas_v)
            else:
                ejes, vista, nota_ejes = (None, "-", "No se detecto vehiculo")
            t_m3 = (time.perf_counter() - t_m3_0) * 1000

            vehiculo_out = None
            if vehiculo_principal is not None:
                vehiculo_out = {
                    "categoria": vehiculo_principal["clase"],
                    "confianza": round(vehiculo_principal["conf"], 3),
                    "area_mascara": vehiculo_principal["area_mascara"],
                    "bbox": [round(v, 1) for v in vehiculo_principal["box"]],
                }

            keypoints_out = [
                [round(float(ll["kps"][0, 0]), 1), round(float(ll["kps"][0, 1]), 1), round(float(ll["kps"][0, 2]), 3)]
                for ll in llantas_v
            ]
            llantas_detalle = [
                {
                    "confianza": round(ll["conf"], 3),
                    "bbox": [round(v, 1) for v in ll["box"]],
                    "centro": [round(float(ll["kps"][0, 0]), 1), round(float(ll["kps"][0, 1]), 1)],
                    "puntos": [[round(float(x), 1), round(float(y), 1), round(float(c), 3)] for x, y, c in ll["kps"]],
                }
                for ll in llantas_v
            ]

            evidencia = (
                self._dibujar_evidencia(img, placa_data, vehiculo_principal, llantas_v, ejes)
                if incluir_imagen
                else None
            )

        total_ms = round((time.perf_counter() - t_total_0) * 1000, 1)
        return {
            "placa": placa_data,
            "vehiculo": vehiculo_out,
            "categoria": vehiculo_out["categoria"] if vehiculo_out else None,
            "confianza": vehiculo_out["confianza"] if vehiculo_out else 0.0,
            "ejes": ejes,
            "vista": vista,
            "nota_ejes": nota_ejes,
            "keypoints": keypoints_out,
            "llantas_detectadas": len(llantas_v),
            "llantas_detalle": llantas_detalle,
            "latencia_ms": {
                "total": total_ms,
                "m1_placa": round(t_m1_det, 1),
                "m1_ocr": round(t_ocr, 1),
                "m2_seg": round(t_m2, 1),
                "m3_pose": round(t_m3, 1),
            },
            "imagen_anotada": evidencia,
        }
