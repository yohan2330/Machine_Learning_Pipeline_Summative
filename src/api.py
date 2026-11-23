
"""FastAPI backend for the malaria detection application."""

import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import List
from urllib.parse import quote

import random

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src import model as model_module
from src import prediction
from src.preprocessing import (
    ALLOWED_EXTENSIONS,
    BASE_DIR,
    CLASSES,
    TEST_DIR,
    TRAIN_DIR,
    UPLOAD_DIR,
    dataset_counts,
    ensure_folders,
    extract_features,
    is_valid_image,
    list_images,
)

STATIC_DIR = os.path.join(BASE_DIR, "static")

START_TIME = time.time()

RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "50"))
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "3"))
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "5"))
AUTO_RETRAIN = os.getenv("AUTO_RETRAIN", "false").lower() == "true"


# ------------------------------------------------------------------
# Training job state
# ------------------------------------------------------------------

training = {
    "status": "idle",
    "kind": None,
    "started_at": None,
    "finished_at": None,
    "message": "No training job yet",
    "result": None,
}

_training_lock = threading.Lock()


# ------------------------------------------------------------------
# Evaluation job state
# ------------------------------------------------------------------

evaluation = {
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "message": "No evaluation yet",
    "result": None,
}

_evaluation_lock = threading.Lock()


# ------------------------------------------------------------------
# Application lifespan
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_folders()
    prediction.load_model()
    yield


app = FastAPI(
    title="Malaria Detection API",
    version="1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Static files and homepage
# ------------------------------------------------------------------

ensure_folders()

app.mount(
    "/train-images",
    StaticFiles(directory=TRAIN_DIR),
    name="train-images",
)

if os.path.isdir(STATIC_DIR):
    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR),
        name="static",
    )


@app.get("/", include_in_schema=False)
def home():
    index = os.path.join(STATIC_DIR, "index.html")

    if not os.path.exists(index):
        raise HTTPException(
            status_code=404,
            detail="static/index.html not found. See /docs for the API.",
        )

    return FileResponse(index)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _pending_uploads() -> int:
    return sum(dataset_counts(UPLOAD_DIR).values())


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------
# Training background job
# ------------------------------------------------------------------

def _run_job(kind: str):
    """Run training or retraining in the background."""

    try:
        if kind == "train":
            result = model_module.train_from_scratch(
                epochs=TRAIN_EPOCHS
            )
        else:
            result = model_module.retrain(
                epochs=RETRAIN_EPOCHS
            )

        prediction.load_model()

        training.update(
            status="done",
            finished_at=_now(),
            message=f"{kind} finished",
            result=result,
        )

    except Exception as e:
        training.update(
            status="failed",
            finished_at=_now(),
            message=str(e),
            result=None,
        )

    finally:
        _training_lock.release()


def _start_job(background_tasks: BackgroundTasks, kind: str):
    """Start a training job if no other training job is running."""

    if not _training_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="A training job is already running.",
        )

    training.update(
        status="running",
        kind=kind,
        started_at=_now(),
        finished_at=None,
        message=f"{kind} in progress...",
        result=None,
    )

    background_tasks.add_task(_run_job, kind)


# ------------------------------------------------------------------
# Evaluation background job
# ------------------------------------------------------------------

def _run_evaluation():
    """Evaluate the model in the background."""

    try:
        model = prediction.get_model()

        if model is None:
            raise ValueError("Model not loaded.")

        metrics = model_module.evaluate_model(model)

        if not metrics:
            raise ValueError(
                "No test images found in data/test."
            )

        metrics["evaluated_at"] = _now()

        info = model_module.load_model_info()
        info["last_evaluation"] = metrics

        model_module.save_model_info(info)

        evaluation.update(
            status="done",
            finished_at=_now(),
            message="Evaluation completed successfully.",
            result=metrics,
        )

    except Exception as e:
        evaluation.update(
            status="failed",
            finished_at=_now(),
            message=str(e),
            result=None,
        )

    finally:
        _evaluation_lock.release()


# ------------------------------------------------------------------
# Health and status
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "online",
        "model_loaded": prediction.is_model_loaded(),
        "uptime_seconds": round(time.time() - START_TIME),
        "started_at": time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(START_TIME),
        ),
    }


@app.get("/status")
def status():
    pending = _pending_uploads()

    return {
        "model_loaded": prediction.is_model_loaded(),
        "model_info": model_module.load_model_info(),
        "pending_uploads": pending,
        "retrain_threshold": RETRAIN_THRESHOLD,
        "retrain_needed": pending >= RETRAIN_THRESHOLD,
        "training": training,
        "evaluation": evaluation,
    }


# ------------------------------------------------------------------
# Dataset endpoints
# ------------------------------------------------------------------

@app.get("/dataset-stats")
def dataset_stats():
    return {
        "train": dataset_counts(TRAIN_DIR),
        "test": dataset_counts(TEST_DIR),
        "uploads": dataset_counts(UPLOAD_DIR),
        "classes": CLASSES,
    }


_features_cache = {}


@app.get("/dataset-features")
def dataset_features(n: int = 200):
    n = max(10, min(n, 1000))

    key = (
        n,
        tuple(dataset_counts(TRAIN_DIR).values()),
    )

    if key not in _features_cache:
        _features_cache.clear()
        _features_cache[key] = extract_features(n)

    return {"rows": _features_cache[key]}


@app.get("/dataset-samples")
def dataset_samples(n: int = 5):
    out = {}

    for c in CLASSES:
        files = list_images(os.path.join(TRAIN_DIR, c))
        random.Random(1).shuffle(files)

        out[c] = [
            f"/train-images/{c}/{quote(f)}"
            for f in files[:max(1, min(n, 10))]
        ]

    return out


# ------------------------------------------------------------------
# Evaluation endpoints
# ------------------------------------------------------------------

@app.post("/evaluate")
def evaluate(background_tasks: BackgroundTasks):

    if not prediction.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded.",
        )

    if sum(dataset_counts(TEST_DIR).values()) == 0:
        raise HTTPException(
            status_code=400,
            detail="No test images in data/test.",
        )

    if not _evaluation_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="An evaluation is already running.",
        )

    evaluation.update(
        status="running",
        started_at=_now(),
        finished_at=None,
        message="Evaluation in progress...",
        result=None,
    )

    background_tasks.add_task(_run_evaluation)

    return {
        "message": "Evaluation started",
        "evaluation": evaluation,
    }


@app.get("/evaluation-status")
def evaluation_status():
    return evaluation


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not prediction.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first.",
        )

    data = await file.read()

    if not is_valid_image(data):
        raise HTTPException(
            status_code=400,
            detail="This file is not a valid image.",
        )

    start = time.time()

    result = prediction.predict_bytes(data)

    result["latency_ms"] = round(
        (time.time() - start) * 1000,
        1,
    )

    return result


# ------------------------------------------------------------------
# Upload
# ------------------------------------------------------------------

@app.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,
    label: str = Form(...),
    files: List[UploadFile] = File(...),
):

    if label not in CLASSES:
        raise HTTPException(
            status_code=400,
            detail=f"label must be one of {CLASSES}",
        )

    ensure_folders()

    saved = 0
    rejected = 0

    for f in files:
        data = await f.read()

        ext = os.path.splitext(f.filename or "")[1].lower()

        if (
            ext not in ALLOWED_EXTENSIONS
            or not is_valid_image(data)
        ):
            rejected += 1
            continue

        path = os.path.join(
            UPLOAD_DIR,
            label,
            f"{uuid.uuid4().hex}{ext}",
        )

        with open(path, "wb") as out:
            out.write(data)

        saved += 1

    pending = _pending_uploads()
    auto_started = False

    if (
        AUTO_RETRAIN
        and pending >= RETRAIN_THRESHOLD
        and not _training_lock.locked()
    ):
        _start_job(background_tasks, "retrain")
        auto_started = True

    return {
        "saved": saved,
        "rejected": rejected,
        "pending_uploads": pending,
        "retrain_needed": pending >= RETRAIN_THRESHOLD,
        "auto_retraining_started": auto_started,
    }


# ------------------------------------------------------------------
# Retraining
# ------------------------------------------------------------------

@app.post("/retrain")
def retrain(background_tasks: BackgroundTasks):

    if _pending_uploads() == 0:
        raise HTTPException(
            status_code=400,
            detail="No uploaded images. Upload data first.",
        )

    if not prediction.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="No model to retrain. Train a model first.",
        )

    _start_job(background_tasks, "retrain")

    return {
        "message": "Retraining started",
        "training": training,
    }


# ------------------------------------------------------------------
# Training from scratch
# ------------------------------------------------------------------

@app.post("/train")
def train(background_tasks: BackgroundTasks):

    if sum(dataset_counts(TRAIN_DIR).values()) == 0:
        raise HTTPException(
            status_code=400,
            detail="No training images in data/train.",
        )

    _start_job(background_tasks, "train")

    return {
        "message": "Training started",
        "training": training,
    }
