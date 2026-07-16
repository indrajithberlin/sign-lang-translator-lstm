import os
import base64
import asyncio
import numpy as np
import cv2
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from inference_engine import InferenceEngine

engine: InferenceEngine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = InferenceEngine(checkpoint_path="checkpoints/best_model.pth")
    engine.start()    # ← also add this so the engine starts properly
    yield
    engine.stop()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static",    exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict")
async def predict(request: Request):
    data       = await request.json()
    session_id = data.get("session_id", "default")
    frame_b64  = data.get("frame", "")

    if not frame_b64:
        return JSONResponse({"error": "missing frame"}, status_code=400)

    # Decode base64 JPEG → BGR numpy frame
    img_bytes = base64.b64decode(frame_b64)
    arr   = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "invalid image"}, status_code=400)

    # Run inference — same pipeline as infer_realtime.py
    label, conf = engine.predict_frame(frame, session_id)

    return JSONResponse({
        "prediction": label,
        "confidence": round(float(conf), 2)
    })

@app.get("/status")
async def get_status():
    return JSONResponse({"status": "ok", "labels": engine.labels})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000,
                ssl_keyfile="server.key",
                ssl_certfile="server.crt")