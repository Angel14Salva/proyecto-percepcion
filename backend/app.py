import os
import io

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from starlette.concurrency import run_in_threadpool

# ==========================
# Config
# ==========================
MODEL_PATH = "best.onnx"
INPUT_SIZE = 320
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45

CLASSES = {
    0: "batería",
    1: "residuo orgánico",
    2: "cartón",
    3: "ropa",
    4: "vidrio",
    5: "metal",
    6: "papel",
    7: "plástico",
    8: "residuo sanitario / cepillo de dientes",
    9: "zapatos",
}

# ==========================
# FastAPI
# ==========================
app = FastAPI(title="Clasificador Inteligente de Residuos")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Modelo ONNX (sesión optimizada para CPU)
# ==========================
print("Cargando modelo ONNX...")

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
# Render free/starter suele dar 1-2 vCPU; usamos todos los hilos disponibles
sess_options.intra_op_num_threads = max(os.cpu_count() or 1, 1)
sess_options.enable_mem_pattern = True
sess_options.enable_cpu_mem_arena = True

session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=sess_options,
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name

print("Modelo ONNX cargado correctamente.")


# ==========================
# NMS (vectorizado)
# ==========================
def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = NMS_THRESHOLD):
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < threshold]

    return keep


# ==========================
# Preprocesamiento (letterbox: mantiene proporción, rellena con gris)
# ==========================
LETTERBOX_COLOR = (114, 114, 114)  # gris estándar de YOLO


def letterbox(image: Image.Image, size: int = INPUT_SIZE):
    w, h = image.size
    scale = min(size / w, size / h)
    new_w, new_h = round(w * scale), round(h * scale)

    resized = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), LETTERBOX_COLOR)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    return canvas, scale, pad_x, pad_y


def preprocess(canvas: Image.Image) -> np.ndarray:
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = np.ascontiguousarray(arr[np.newaxis, ...])
    return arr


# ==========================
# NMS por clase (evita que una detección de una clase suprima
# a una detección real de otra clase solo por solaparse)
# ==========================
def nms_per_class(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, threshold: float = NMS_THRESHOLD):
    keep_all = []
    for cid in np.unique(class_ids):
        idxs = np.where(class_ids == cid)[0]
        keep = nms(boxes[idxs], scores[idxs], threshold)
        keep_all.extend(idxs[keep].tolist())
    return keep_all


# ==========================
# Inferencia (vectorizada, sin loops en Python sobre los 2100 anchors)
# ==========================
def run_inference(image: Image.Image):
    ancho_original, alto_original = image.size

    canvas, scale, pad_x, pad_y = letterbox(image, INPUT_SIZE)
    tensor = preprocess(canvas)
    outputs = session.run(None, {input_name: tensor})

    # outputs[0]: (1, 14, 2100) -> (2100, 14)
    predictions = outputs[0][0].transpose()

    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]

    mask = confidences > CONF_THRESHOLD
    if not np.any(mask):
        return []

    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]

    # Coordenadas en el canvas 320x320 con letterbox -> quitamos el padding
    # y des-escalamos al tamaño real de la imagen original.
    x1 = (cx - w / 2 - pad_x) / scale
    y1 = (cy - h / 2 - pad_y) / scale
    x2 = (cx + w / 2 - pad_x) / scale
    y2 = (cy + h / 2 - pad_y) / scale

    x1 = np.clip(x1, 0, ancho_original)
    y1 = np.clip(y1, 0, alto_original)
    x2 = np.clip(x2, 0, ancho_original)
    y2 = np.clip(y2, 0, alto_original)

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms_per_class(boxes, confidences, class_ids, NMS_THRESHOLD)

    detecciones = []
    for i in keep:
        detecciones.append(
            {
                "class": CLASSES[int(class_ids[i])],
                "confidence": round(float(confidences[i]), 3),
                "bbox": [round(float(v), 1) for v in boxes[i]],
            }
        )

    return detecciones


# ==========================
# Rutas
# ==========================
@app.get("/")
def inicio():
    return {"mensaje": "API de Clasificación de Residuos funcionando correctamente."}


@app.get("/health")
def health():
    return {"status": "ok", "modelo": "YOLO11n ONNX", "estado": "activo"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contenido = await file.read()
    imagen = Image.open(io.BytesIO(contenido)).convert("RGB")

    # La inferencia es CPU-bound y bloqueante: la mandamos a un threadpool
    # para no congelar el event loop de FastAPI mientras corre.
    detecciones = await run_in_threadpool(run_inference, imagen)

    return {"detections": detecciones}
