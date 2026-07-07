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
    0: "battery",
    1: "biological",
    2: "cardboard",
    3: "clothes",
    4: "glass",
    5: "metal",
    6: "paper",
    7: "plastic",
    8: "sanitary waste and toothbrushes",
    9: "shoes",
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
# Preprocesamiento
# ==========================
def preprocess(image: Image.Image) -> np.ndarray:
    # BILINEAR es notablemente más rápido que el LANCZOS por defecto de PIL
    # y para un modelo de 320x320 en tiempo real la diferencia de calidad es imperceptible.
    image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)

    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = np.ascontiguousarray(arr[np.newaxis, ...])
    return arr


# ==========================
# Inferencia (vectorizada, sin loops en Python sobre los 2100 anchors)
# ==========================
def run_inference(image: Image.Image):
    ancho_original, alto_original = image.size

    tensor = preprocess(image)
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

    scale_x = ancho_original / INPUT_SIZE
    scale_y = alto_original / INPUT_SIZE

    x1 = (cx - w / 2) * scale_x
    y1 = (cy - h / 2) * scale_y
    x2 = (cx + w / 2) * scale_x
    y2 = (cy + h / 2) * scale_y

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms(boxes, confidences, NMS_THRESHOLD)

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
