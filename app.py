#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import uuid
import datetime as dt
from typing import Tuple, Dict, Any, List

from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2

# =========================
#   Configuración BD
# =========================
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "password")
DB_NAME = os.getenv("DB_NAME", "detecciones_db")

USE_MYSQL = True
try:
    import mysql.connector
except Exception as e:
    USE_MYSQL = False
    print("[WARN] mysql-connector-python no disponible. La API funcionará sin insertar en MySQL. "
          "Instala el paquete y configura variables de entorno.")

# =========================
#   YOLO (Ultralytics)
# =========================
from ultralytics import YOLO
MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8n.pt")
yolo_model = YOLO(MODEL_NAME)

# =========================
#   Flask + rutas estáticas
# =========================
app = Flask(__name__)
BASE_UPLOAD = os.path.join("static", "uploads")
UPLOAD_IMG = os.path.join(BASE_UPLOAD, "images")
UPLOAD_VID = os.path.join(BASE_UPLOAD, "videos")
UPLOAD_SNAP = os.path.join(BASE_UPLOAD, "snapshots")
for d in (BASE_UPLOAD, UPLOAD_IMG, UPLOAD_VID, UPLOAD_SNAP):
    os.makedirs(d, exist_ok=True)

# =========================
#   Utilidades
# =========================
def bgr_to_hex(bgr: Tuple[int, int, int]) -> str:
    b, g, r = [int(x) for x in bgr]
    return "#%02x%02x%02x" % (r, g, b)

def dominant_color_kmeans(bgr_roi: np.ndarray, K: int = 3) -> Tuple[str, Tuple[int, int, int]]:
    """Color dominante en ROI (BGR) usando k-means. Devuelve (#hex, (b,g,r))."""
    Z = bgr_roi.reshape((-1, 3)).astype(np.float32)
    if Z.size == 0:
        return "#000000", (0, 0, 0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    K = max(1, K)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten())
    dom_bgr = centers[counts.argmax()].astype(int)
    hex_color = bgr_to_hex(tuple(dom_bgr))
    return hex_color, tuple(int(x) for x in dom_bgr)

def classify_shape_from_roi(bgr_roi: np.ndarray) -> str:
    """
    Clasifica forma en el ROI: círculo / cuadrado / rectángulo
    - círculo si circularidad >= 0.85
    - cuadrado si ratio ancho/alto ∈ [0.9, 1.1]
    """
    if bgr_roi.size == 0:
        return "rectangulo"
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return "rectangulo"
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True) + 1e-6
    circularity = 4.0 * np.pi * (area / (peri * peri))
    if circularity >= 0.85:
        return "circulo"
    x, y, w, h = cv2.boundingRect(c)
    ratio = w / float(h)
    if 0.9 <= ratio <= 1.1:
        return "cuadrado"
    return "rectangulo"

def connect_db():
    if not USE_MYSQL:
        return None
    return mysql.connector.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, database=DB_NAME
    )

def insert_detection(objeto: str, forma: str, color_hex: str, imagen_url: str) -> int:
    """Inserta y devuelve ID (o -1 si no hay DB)."""
    if not USE_MYSQL:
        print("[LOG] (sin-DB) Guardado simulado:", objeto, forma, color_hex, imagen_url)
        return -1
    cnx = connect_db()
    cur = cnx.cursor()
    now = dt.datetime.now()
    cur.execute(
        "INSERT INTO detecciones (objeto, forma, color, fecha_hora, imagen_url) VALUES (%s, %s, %s, %s, %s)",
        (objeto, forma, color_hex, now, imagen_url),
    )
    cnx.commit()
    last_id = cur.lastrowid
    cur.close(); cnx.close()
    return last_id

def get_last_detection() -> Dict[str, Any]:
    if not USE_MYSQL:
        return {"id": -1, "objeto": "", "forma": "", "color": "", "fecha_hora": "", "imagen_url": ""}
    cnx = connect_db()
    cur = cnx.cursor(dictionary=True)
    cur.execute("SELECT * FROM detecciones ORDER BY id DESC LIMIT 1")
    row = cur.fetchone() or {}
    cur.close(); cnx.close()
    return row

def get_detections(limit: int = 20) -> List[Dict[str, Any]]:
    if not USE_MYSQL:
        return []
    cnx = connect_db()
    cur = cnx.cursor(dictionary=True)
    cur.execute("SELECT * FROM detecciones ORDER BY id DESC LIMIT %s", (limit,))
    rows = cur.fetchall() or []
    cur.close(); cnx.close()
    return rows

def allowed_image(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"png", "jpg", "jpeg", "bmp", "webp"}

def allowed_video(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"mp4", "avi", "mov", "mkv", "webm"}

def load_image_bgr_from_upload(file_storage) -> np.ndarray:
    """Lee la imagen subida con OpenCV (más tolerante que PIL). Devuelve BGR."""
    data = file_storage.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError("Imagen inválida o formato no soportado")
    return img

def _class_name_from_id(cls_id: int, names) -> str:
    if isinstance(names, dict):
        return names.get(cls_id, f"cls_{cls_id}")
    if isinstance(names, list):
        return names[cls_id] if 0 <= cls_id < len(names) else f"cls_{cls_id}"
    return str(cls_id)

def run_yolo_on_image(image_bgr: np.ndarray) -> Dict[str, Any]:
    results = yolo_model(image_bgr, verbose=False)
    if not results or len(results[0].boxes) == 0:
        objeto = "desconocido"
        forma = "rectangulo"
        color_hex, _ = dominant_color_kmeans(image_bgr)
        return {"objeto": objeto, "forma": forma, "color": color_hex, "bbox": None}

    r0 = results[0]
    confs = r0.boxes.conf.cpu().numpy()
    idx = int(np.argmax(confs))
    box = r0.boxes[idx]
    cls_id = int(box.cls.item()) if box.cls is not None else -1
    names = getattr(r0, "names", None) or getattr(yolo_model, "names", None)

    objeto = _class_name_from_id(cls_id, names)
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image_bgr.shape[1] - 1, x2), min(image_bgr.shape[0] - 1, y2)
    roi = image_bgr[y1:y2, x1:x2].copy()

    forma = classify_shape_from_roi(roi)
    color_hex, _ = dominant_color_kmeans(roi, K=3)
    return {"objeto": objeto, "forma": forma, "color": color_hex, "bbox": [int(x1), int(y1), int(x2), int(y2)]}

# =========================
#   Rutas
# =========================
@app.get("/health")
def health():
    return "ok", 200

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/detectar")
def detectar():
    """Sube una IMAGEN, detecta y guarda."""
    try:
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "No se envió archivo"}), 400
        file = request.files["file"]
        if file.filename == "" or not allowed_image(file.filename):
            return jsonify({"ok": False, "error": "Archivo de imagen inválido"}), 400

        uid = dt.datetime.now().strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:8]
        filename = f"{uid}.jpg"
        save_path = os.path.join(UPLOAD_IMG, filename)

        # Lectura robusta con OpenCV
        image_np = load_image_bgr_from_upload(file)
        cv2.imwrite(save_path, image_np)

        det = run_yolo_on_image(image_np)

        rel_url = f"/{save_path.replace(os.sep, '/')}"
        det_id = insert_detection(det["objeto"], det["forma"], det["color"], rel_url)

        return jsonify({"ok": True, "id": det_id, "imagen_url": rel_url, **det})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/detectar_stream")
def detectar_stream():
    """
    Recibe un frame (IMAGEN) desde la cámara web (front manda un JPEG).
    Si llega form 'save=1', guarda snapshot con bbox + inserta en BD.
    """
    try:
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "No se envió frame"}), 400
        file = request.files["file"]
        if file.filename == "" or not allowed_image(file.filename):
            return jsonify({"ok": False, "error": "Frame inválido"}), 400

        image_np = load_image_bgr_from_upload(file)
        det = run_yolo_on_image(image_np)

        rel_url = None
        det_id = -1
        save_flag = (request.form.get("save", "").lower() in ("1", "true", "on"))
        if save_flag:
            uid = dt.datetime.now().strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:8]
            fname = f"{uid}_snap.jpg"
            save_path = os.path.join(UPLOAD_SNAP, fname)
            if det.get("bbox"):
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_np, det["objeto"], (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imwrite(save_path, image_np)
            rel_url = f"/{save_path.replace(os.sep, '/')}"
            det_id = insert_detection(det["objeto"], det["forma"], det["color"], rel_url)

        return jsonify({"ok": True, "id": det_id, "imagen_url": rel_url, **det})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/detectar_video")
def detectar_video():
    """
    Sube un VIDEO. Se muestrean ~1 frame/seg y se guardan hasta 20 snapshots con bbox.
    Cada snapshot insertado queda en BD.
    """
    try:
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "No se envió video"}), 400
        file = request.files["file"]
        if file.filename == "" or not allowed_video(file.filename):
            return jsonify({"ok": False, "error": "Archivo de video inválido"}), 400

        uid = dt.datetime.now().strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:8]
        ext = file.filename.rsplit(".", 1)[1].lower()
        video_name = f"{uid}.{ext}"
        video_path = os.path.join(UPLOAD_VID, video_name)
        file.save(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"ok": False, "error": "No se pudo abrir el video"}), 400

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(fps))  # ~1 frame por segundo
        max_snaps = 20
        count = 0
        detecciones: List[Dict[str, Any]] = []

        i = 0
        while True:
            ret = cap.grab()
            if not ret:
                break
            if i % step == 0:
                ret2, frame = cap.retrieve()
                if not ret2:
                    break
                det = run_yolo_on_image(frame)
                # Dibuja bbox y etiqueta
                if det.get("bbox"):
                    x1, y1, x2, y2 = det["bbox"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, det["objeto"], (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                snap_name = f"{uid}_{count:02d}.jpg"
                snap_path = os.path.join(UPLOAD_SNAP, snap_name)
                cv2.imwrite(snap_path, frame)
                rel_url = f"/{snap_path.replace(os.sep, '/')}"
                det_id = insert_detection(det["objeto"], det["forma"], det["color"], rel_url)
                detecciones.append({"id": det_id, "imagen_url": rel_url, **det})
                count += 1
                if count >= max_snaps:
                    break
            i += 1
        cap.release()

        return jsonify({
            "ok": True,
            "video_guardado": f"/{video_path.replace(os.sep, '/')}",
            "frames_totales": total,
            "fps": fps,
            "snapshots_guardados": count,
            "detecciones": detecciones
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/api/ultimo")
def api_ultimo():
    row = get_last_detection()
    return jsonify(row or {})

@app.get("/api/detecciones")
def api_detecciones():
    try:
        limit = int(request.args.get("limit", "20"))
        rows = get_detections(limit=limit)
        return jsonify(rows)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# =========================
#   Main
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)
