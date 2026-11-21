"""Prediction: load the saved model and predict one image."""
import os
import threading

from tensorflow import keras

from src.preprocessing import CLASSES, MODEL_PATH, load_image_from_bytes

_model = None
_lock = threading.Lock()


def load_model():
    """Load the model from models/ into memory. Returns True if it worked."""
    global _model
    if not os.path.exists(MODEL_PATH):
        return False
    new_model = keras.models.load_model(MODEL_PATH, compile=False)
    with _lock:
        _model = new_model
    return True


def is_model_loaded() -> bool:
    return _model is not None


def predict_bytes(data: bytes) -> dict:
    """Predict one image (given as bytes). Returns the label and the confidence."""
    with _lock:
        model = _model          # take a reference (the model may be swapped after retraining)
    if model is None:
        raise RuntimeError("Model not loaded")
    x = load_image_from_bytes(data)
    # Calling the model directly is faster than model.predict() for one image
    p = float(model(x, training=False).numpy()[0][0])   # probability of "Uninfected"
    index = int(p >= 0.5)
    return {
        "label": CLASSES[index],
        "confidence": round(p if index == 1 else 1 - p, 4),
        "probabilities": {CLASSES[0]: round(1 - p, 4), CLASSES[1]: round(p, 4)},
    }
