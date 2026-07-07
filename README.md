# Clasificador Inteligente de Residuos

Detección de residuos en tiempo real desde webcam, usando YOLO11n exportado a ONNX
y servido con FastAPI + ONNX Runtime (CPU).

## Estructura

```
backend/     API FastAPI + modelo ONNX
frontend/    Cliente web (webcam + canvas de detecciones)
render.yaml  Blueprint para desplegar ambos servicios en Render
```

## Correr en local

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Abre `frontend/index.html` con Live Server (o cualquier servidor estático) y cambia
`API` en `frontend/script.js` a `http://127.0.0.1:8000/predict`.

## Deploy en Render

### Opción A: Blueprint (recomendado, un solo paso)
1. En Render, "New" → "Blueprint" → conecta este repo.
2. Render detecta `render.yaml` y crea los dos servicios (backend Docker + frontend estático).
3. Cuando termine, copia la URL del backend y pégala en `frontend/script.js` (constante `API`),
   luego vuelve a hacer push — Render redeploya solo (`autoDeploy: true`).

### Opción B: Manual
1. **Backend**: New → Web Service → conecta el repo → Root Directory: `backend` →
   Runtime: Docker → Health Check Path: `/health`.
2. **Frontend**: New → Static Site → conecta el repo → Root Directory: `frontend` →
   Build Command: (vacío) → Publish Directory: `.`.

## Sobre la velocidad de detección

El cuello de botella real no era la red, era:
1. Un `for` en Python puro recorriendo los ~2100 anchors del output del modelo en
   cada frame → reemplazado por operaciones vectorizadas con numpy.
2. La sesión de ONNX Runtime sin optimizaciones de grafo ni hilos configurados →
   ahora usa `ORT_ENABLE_ALL` y todos los hilos de CPU disponibles.
3. El **plan de Render**: en el plan gratuito, el servicio se "duerme" tras ~15 min
   sin tráfico y la primera petición tras eso tarda varios segundos (cold start).
   Si necesitas latencia consistente, usa un plan pago (Starter en adelante) que
   no se duerme. `render.yaml` ya está configurado en `starter`; puedes bajarlo a
   `free` desde el dashboard si prefieres no pagar, sabiendo que tendrás ese
   arranque en frío ocasional.

Con esto, en un plan que no duerma, cada frame de 320x320 debería resolverse en
CPU en el orden de decenas de milisegundos (antes del NMS/red), muy por debajo del
tiempo de round-trip HTTP.
