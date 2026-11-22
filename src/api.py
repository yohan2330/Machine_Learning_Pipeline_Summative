"""FastAPI backend.

Run from the project root:   python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
Interactive docs:            http://localhost:8000/docs
"""
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import List

import random
from urllib.parse import quote

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src import model as model_module
from src import prediction
from src.preprocessing import (ALLOWED_EXTENSIONS, BASE_DIR, CLASSES, TEST_DIR,
                               TRAIN_DIR, UPLOAD_DIR, dataset_counts, ensure_folders,
                               extract_features, is_valid_image, list_images)

STATIC_DIR = os.path.join(BASE_DIR, "static")   # the HTML user interface lives here

START_TIME = time.time()
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "50"))  # retraining trigger
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "3"))
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "5"))
AUTO_RETRAIN = os.getenv("AUTO_RETRAIN", "false").lower() == "true"

# State of the training job (shown in the UI)
training = {"status": "idle", "kind": None, "started_at": None, "finished_at": None,
            "message": "No training job yet", "result": None}
_training_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_folders()
    prediction.load_model()
    yield


app = FastAPI(title="Malaria Detection API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---- User interface (HTML) served by the API itself: http://localhost:8000/
ensure_folders()
app.mount("/train-images", StaticFiles(directory=TRAIN_DIR), name="train-images")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def home():
    index = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index):
        raise HTTPException(status_code=404, detail="static/index.html not found. See /docs for the API.")
    return FileResponse(index)


def _pending_uploads() -> int:
    return sum(dataset_counts(UPLOAD_DIR).values())


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _run_job(kind: str):
    """Runs in the background so the API keeps answering during training."""
    try:
        if kind == "train":
            result = model_module.train_from_scratch(epochs=TRAIN_EPOCHS)
        else:
            result = model_module.retrain(epochs=RETRAIN_EPOCHS)
        prediction.load_model()          # switch to the new model
        training.update(status="done", finished_at=_now(),
                        message=f"{kind} finished", result=result)
    except Exception as e:               # keep the error visible in the UI
        training.update(status="failed", finished_at=_now(), message=str(e), result=None)
    finally:
        _training_lock.release()


def _start_job(background_tasks: BackgroundTasks, kind: str):
    if not _training_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A training job is already running")
    training.update(status="running", kind=kind, started_at=_now(), finished_at=None,
                    message=f"{kind} in progress...", result=None)
    background_tasks.add_task(_run_job, kind)


# ---------------------------------------------------------------- health / status
@app.get("/health")
def health():
    """Model up-time: is the API alive, is the model loaded, since when."""
    return {
        "status": "online",
        "model_loaded": prediction.is_model_loaded(),
        "uptime_seconds": round(time.time() - START_TIME),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(START_TIME)),
    }


@app.get("/status")
def status():
    """Model info + training state + whether a retraining is needed."""
    pending = _pending_uploads()
    return {
        "model_loaded": prediction.is_model_loaded(),
        "model_info": model_module.load_model_info(),
        "pending_uploads": pending,
        "retrain_threshold": RETRAIN_THRESHOLD,
        "retrain_needed": pending >= RETRAIN_THRESHOLD,
        "training": training,
    }


@app.get("/dataset-stats")
def dataset_stats():
    """Number of images per class (used for the visualizations)."""
    return {"train": dataset_counts(TRAIN_DIR), "test": dataset_counts(TEST_DIR),
            "uploads": dataset_counts(UPLOAD_DIR), "classes": CLASSES}


_features_cache = {}


@app.get("/dataset-features")
def dataset_features(n: int = 200):
    """3 simple features for up to n images per class (used for the charts)."""
    n = max(10, min(n, 1000))
    key = (n, tuple(dataset_counts(TRAIN_DIR).values()))
    if key not in _features_cache:
        _features_cache.clear()
        _features_cache[key] = extract_features(n)
    return {"rows": _features_cache[key]}


@app.get("/dataset-samples")
def dataset_samples(n: int = 5):
    """A few example training images per class (URLs)."""
    out = {}
    for c in CLASSES:
        files = list_images(os.path.join(TRAIN_DIR, c))
        random.Random(1).shuffle(files)
        out[c] = [f"/train-images/{c}/{quote(f)}" for f in files[:max(1, min(n, 10))]]
    return out


@app.post("/evaluate")
def evaluate():
    """Evaluate the model that is currently serving predictions on data/test."""
    model = prediction.get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if sum(dataset_counts(TEST_DIR).values()) == 0:
        raise HTTPException(status_code=400, detail="No test images in data/test.")
    metrics = model_module.evaluate_model(model)
    metrics["evaluated_at"] = _now()
    info = model_module.load_model_info()
    info["last_evaluation"] = metrics
    model_module.save_model_info(info)
    return metrics


# ---------------------------------------------------------------- prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict ONE image."""
    if not prediction.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")
    data = await file.read()
    if not is_valid_image(data):
        raise HTTPException(status_code=400, detail="This file is not a valid image")
    start = time.time()
    result = prediction.predict_bytes(data)
    result["latency_ms"] = round((time.time() - start) * 1000, 1)
    return result


# ---------------------------------------------------------------- upload + train / retrain
@app.post("/upload")
async def upload(background_tasks: BackgroundTasks,
                 label: str = Form(...),
                 files: List[UploadFile] = File(...)):
    """Upload MANY images of one class (label) for the next retraining."""
    if label not in CLASSES:
        raise HTTPException(status_code=400, detail=f"label must be one of {CLASSES}")
    ensure_folders()
    saved, rejected = 0, 0
    for f in files:
        data = await f.read()
        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS or not is_valid_image(data):
            rejected += 1
            continue
        path = os.path.join(UPLOAD_DIR, label, f"{uuid.uuid4().hex}{ext}")
        with open(path, "wb") as out:
            out.write(data)
        saved += 1

    pending = _pending_uploads()
    auto_started = False
    if AUTO_RETRAIN and pending >= RETRAIN_THRESHOLD and not _training_lock.locked():
        _start_job(background_tasks, "retrain")       # automatic trigger
        auto_started = True
    return {"saved": saved, "rejected": rejected, "pending_uploads": pending,
            "retrain_needed": pending >= RETRAIN_THRESHOLD,
            "auto_retraining_started": auto_started}


@app.post("/retrain")
def retrain(background_tasks: BackgroundTasks):
    """Manual trigger (the UI button): retrain with the uploaded images."""
    if _pending_uploads() == 0:
        raise HTTPException(status_code=400, detail="No uploaded images. Upload data first.")
    if not prediction.is_model_loaded():
        raise HTTPException(status_code=503, detail="No model to retrain. Train a model first.")
    _start_job(background_tasks, "retrain")
    return {"message": "Retraining started", "training": training}


@app.post("/train")
def train(background_tasks: BackgroundTasks):
    """Train a brand new model from scratch on data/train (runs in the background)."""
    if sum(dataset_counts(TRAIN_DIR).values()) == 0:
        raise HTTPException(status_code=400, detail="No training images in data/train.")
    _start_job(background_tasks, "train")
    return {"message": "Training started", "training": training}
