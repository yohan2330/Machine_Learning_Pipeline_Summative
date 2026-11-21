"""FastAPI backend.

Run from the project root:   uvicorn src.api:app --host 0.0.0.0 --port 8000
Interactive docs:            http://localhost:8000/docs
"""
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import List

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src import model as model_module
from src import prediction
from src.preprocessing import (ALLOWED_EXTENSIONS, CLASSES, TEST_DIR, TRAIN_DIR,
                               UPLOAD_DIR, dataset_counts, ensure_folders,
                               is_valid_image)

START_TIME = time.time()
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "50"))  # retraining trigger
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "3"))
AUTO_RETRAIN = os.getenv("AUTO_RETRAIN", "false").lower() == "true"

# State of the retraining job (shown in the UI)
training = {"status": "idle", "started_at": None, "finished_at": None,
            "message": "No retraining yet", "result": None}
_training_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_folders()
    prediction.load_model()
    yield


app = FastAPI(title="Malaria Detection API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _pending_uploads() -> int:
    return sum(dataset_counts(UPLOAD_DIR).values())


def _run_retraining():
    """Runs in the background so the API keeps answering during training."""
    try:
        result = model_module.retrain(epochs=RETRAIN_EPOCHS)
        prediction.load_model()          # switch to the new model
        training.update(status="done", finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                        message="Retraining finished", result=result)
    except Exception as e:               # keep the error visible in the UI
        training.update(status="failed", finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                        message=str(e), result=None)
    finally:
        _training_lock.release()


def _start_retraining(background_tasks: BackgroundTasks):
    if not _training_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A retraining is already running")
    training.update(status="running", started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    finished_at=None, message="Retraining in progress...", result=None)
    background_tasks.add_task(_run_retraining)


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
    """Model info + retraining state + whether a retraining is needed."""
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


# ---------------------------------------------------------------- upload + retrain
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
        _start_retraining(background_tasks)       # automatic trigger
        auto_started = True
    return {"saved": saved, "rejected": rejected, "pending_uploads": pending,
            "retrain_needed": pending >= RETRAIN_THRESHOLD,
            "auto_retraining_started": auto_started}


@app.post("/retrain")
def retrain(background_tasks: BackgroundTasks):
    """Manual trigger (the UI button). Training runs in the background."""
    if _pending_uploads() == 0:
        raise HTTPException(status_code=400, detail="No uploaded images. Upload data first.")
    _start_retraining(background_tasks)
    return {"message": "Retraining started", "training": training}
