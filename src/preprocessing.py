"""Preprocessing: paths, constants and image loading.

Every image goes through the same steps as in the notebook:
convert to RGB -> resize to 128x128. The pixel scaling (-1 to 1) is done
inside the model itself (Rescaling layer), so we keep raw 0-255 values here.
"""
import io
import os
import random

import numpy as np
from PIL import Image

# ----- Constants -----
CLASSES = ["Parasitized", "Uninfected"]   # index 0 and 1 (same order as the notebook)
IMG_SIZE = (128, 128)
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# ----- Paths (project root = parent of src/) -----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")   # new data waiting for retraining
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "malaria_model.h5")
MODEL_INFO_PATH = os.path.join(MODELS_DIR, "model_info.json")


def ensure_folders():
    """Create the data folders if they do not exist yet."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    for base in (TRAIN_DIR, TEST_DIR, UPLOAD_DIR):
        for c in CLASSES:
            os.makedirs(os.path.join(base, c), exist_ok=True)


def is_valid_image(data: bytes) -> bool:
    """True if the bytes are a real image that can be opened."""
    try:
        Image.open(io.BytesIO(data)).verify()
        return True
    except Exception:
        return False


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Bytes of an image -> array of shape (1, 128, 128, 3), ready for the model."""
    img = Image.open(io.BytesIO(data)).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    return np.expand_dims(np.array(img, dtype="float32"), axis=0)


def count_images(folder: str) -> int:
    """Count the images inside a folder (not recursive)."""
    if not os.path.isdir(folder):
        return 0
    return sum(1 for f in os.listdir(folder)
               if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS)


def dataset_counts(base_dir: str) -> dict:
    """Number of images per class for data/train, data/test or data/uploads."""
    return {c: count_images(os.path.join(base_dir, c)) for c in CLASSES}


def list_images(folder: str) -> list:
    """Sorted image file names of a folder."""
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder)
                  if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS)


def extract_features(n_per_class: int = 200, seed: int = 42) -> list:
    """3 simple features per image, used for the visualizations:
    average green value, contrast (std of gray levels) and image size (pixels)."""
    rng = random.Random(seed)
    rows = []
    for c in CLASSES:
        folder = os.path.join(TRAIN_DIR, c)
        files = list_images(folder)
        rng.shuffle(files)
        for f in files[:n_per_class]:
            try:
                img = Image.open(os.path.join(folder, f)).convert("RGB")
            except Exception:
                continue
            w, h = img.size
            arr = np.asarray(img.resize((64, 64)), dtype="float32")
            rows.append({"class": c,
                         "mean_green": round(float(arr[:, :, 1].mean()), 2),
                         "contrast": round(float(arr.mean(axis=2).std()), 2),
                         "area_px": w * h})
    return rows
