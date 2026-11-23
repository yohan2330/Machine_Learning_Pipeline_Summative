
"""Model: build, train, evaluate and RETRAIN the malaria classifier."""

import gc
import json
import os
import shutil
import time

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from tensorflow import keras
from tensorflow.keras import layers

from src.preprocessing import (
    ALLOWED_EXTENSIONS,
    CLASSES,
    IMG_SIZE,
    MODEL_INFO_PATH,
    MODEL_PATH,
    TEST_DIR,
    TRAIN_DIR,
    UPLOAD_DIR,
    dataset_counts,
    ensure_folders,
    list_images,
    load_image_from_bytes,
)

BATCH_SIZE = 32
SEED = 42
NEW_DATA_REPEAT = 3


def build_model():
    """Build MobileNetV2 transfer learning model."""

    base = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )

    base.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))

    x = layers.RandomFlip(
        "horizontal_and_vertical"
    )(inputs)

    x = layers.Rescaling(
        1 / 127.5,
        offset=-1,
    )(x)

    x = base(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
    )(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def make_dataset(directory, shuffle=True):
    """Create a batched TensorFlow dataset."""

    return keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        class_names=CLASSES,
        shuffle=shuffle,
        seed=SEED,
    )


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_model(model) -> dict:
    """Evaluate the model using small batches to reduce memory usage."""

    if sum(dataset_counts(TEST_DIR).values()) == 0:
        return {}

    files = []

    for label, cls in enumerate(CLASSES):

        folder = os.path.join(TEST_DIR, cls)

        for name in list_images(folder):
            files.append(
                (os.path.join(folder, name), label)
            )

    y_true = []
    y_pred = []

    # Smaller batches reduce peak memory usage on Render.
    group = 2

    for i in range(0, len(files), group):

        chunk = files[i:i + group]

        batch = []

        for path, _ in chunk:

            with open(path, "rb") as f:
                image = load_image_from_bytes(f.read())[0]

            batch.append(image)

        probs = model(
            np.stack(batch),
            training=False,
        ).numpy().ravel()

        y_pred.extend(
            (probs >= 0.5).astype(int).tolist()
        )

        y_true.extend(
            label for _, label in chunk
        )

        del batch
        del probs

        # Release unused Python memory periodically.
        if (i // group + 1) % 8 == 0:
            gc.collect()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "accuracy": round(
            float(accuracy_score(y_true, y_pred)), 4
        ),

        "precision": round(
            float(precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )), 4
        ),

        "recall": round(
            float(recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )), 4
        ),

        "f1": round(
            float(f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )), 4
        ),

        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).tolist(),

        "test_images": int(len(y_true)),
    }

    gc.collect()

    return metrics


# ------------------------------------------------------------------
# Model information
# ------------------------------------------------------------------

def load_model_info() -> dict:

    if os.path.exists(MODEL_INFO_PATH):

        with open(MODEL_INFO_PATH) as f:
            return json.load(f)

    return {}


def save_model_info(info: dict):

    with open(MODEL_INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)


def _backup_and_save(model):
    """Back up the previous model before saving the new one."""

    if os.path.exists(MODEL_PATH):

        shutil.copy(
            MODEL_PATH,
            MODEL_PATH.replace(
                ".h5",
                "_previous.h5",
            ),
        )

    model.save(MODEL_PATH)


# ------------------------------------------------------------------
# Train from scratch
# ------------------------------------------------------------------

def train_from_scratch(epochs=5) -> dict:

    ensure_folders()

    if sum(dataset_counts(TRAIN_DIR).values()) == 0:
        raise ValueError(
            "No training images found in data/train."
        )

    start = time.time()

    model = build_model()

    history = model.fit(
        make_dataset(TRAIN_DIR),
        epochs=epochs,
        verbose=2,
    )

    metrics = evaluate_model(model)

    _backup_and_save(model)

    info = {
        "mode": "train from scratch",

        "last_trained": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "training_images": sum(
            dataset_counts(TRAIN_DIR).values()
        ),

        "new_images_used": 0,

        "epochs": epochs,

        "training_seconds": round(
            time.time() - start,
            1,
        ),

        "final_train_accuracy": round(
            float(history.history["accuracy"][-1]),
            4,
        ),

        "test_metrics": metrics,
    }

    save_model_info(info)

    return info


# ------------------------------------------------------------------
# Move uploaded images
# ------------------------------------------------------------------

def _move_uploads_to_train():

    moved = 0

    for c in CLASSES:

        src_dir = os.path.join(UPLOAD_DIR, c)
        dst_dir = os.path.join(TRAIN_DIR, c)

        os.makedirs(dst_dir, exist_ok=True)

        for f in os.listdir(src_dir):

            if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS:

                shutil.move(
                    os.path.join(src_dir, f),
                    os.path.join(dst_dir, f),
                )

                moved += 1

    return moved


# ------------------------------------------------------------------
# Retraining
# ------------------------------------------------------------------

def retrain(epochs=3) -> dict:
    """Retrain the current model using old and newly uploaded images."""

    ensure_folders()

    n_new = sum(dataset_counts(UPLOAD_DIR).values())

    if n_new == 0:
        raise ValueError(
            "No uploaded images to retrain with. Upload data first."
        )

    start = time.time()

    model = keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    old_ds = make_dataset(TRAIN_DIR)

    new_ds = make_dataset(UPLOAD_DIR).repeat(
        NEW_DATA_REPEAT
    )

    history = model.fit(
        old_ds.concatenate(new_ds),
        epochs=epochs,
        verbose=2,
    )

    metrics = evaluate_model(model)

    _backup_and_save(model)

    moved = _move_uploads_to_train()

    info = {
        "mode": "retrain",

        "last_trained": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "training_images": sum(
            dataset_counts(TRAIN_DIR).values()
        ),

        "new_images_used": moved,

        "epochs": epochs,

        "training_seconds": round(
            time.time() - start,
            1,
        ),

        "final_train_accuracy": round(
            float(history.history["accuracy"][-1]),
            4,
        ),

        "test_metrics": metrics,
    }

    save_model_info(info)

    return info
