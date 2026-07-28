"""Champion model lifecycle and single-image inference.

The model is loaded once at process start and held in a module-level
reference. Loading per request would dominate latency and invalidate the
load-test results.
"""
import threading
import time

import numpy as np
import tensorflow as tf

from . import config, database, preprocessing

_model = None
_version = "none"
_lock = threading.Lock()


def set_model(model, version: str) -> None:
    """Swap the served model atomically so in-flight requests are unaffected."""
    global _model, _version
    with _lock:
        _model = model
        _version = version


def get_model():
    return _model


def get_version() -> str:
    return _version


def is_ready() -> bool:
    return _model is not None


def load_champion() -> None:
    """Load the registered champion, falling back to the default path."""
    record = database.get_champion()
    path = config.CHAMPION_PATH
    version = "coffee_model"
    if record and record["path"] and tf.io.gfile.exists(record["path"]):
        path = record["path"]
        version = record["version"]
    model = tf.keras.models.load_model(path)
    set_model(model, version)


def warmup() -> None:
    """Run one inference so the first real request isn't paying graph tracing.

    Without this, p99 latency in the load test is dominated by the very first
    request to each container.
    """
    if _model is None:
        return
    blank = np.zeros((1, *config.IMG_SIZE, 3), dtype="float32")
    _model.predict(blank, verbose=0)


def predict_image(payload: bytes) -> dict:
    with _lock:
        model = _model
        version = _version
    if model is None:
        raise RuntimeError("Model is not loaded")

    started = time.perf_counter()
    image = preprocessing.decode_image_bytes(payload)
    batch = tf.expand_dims(image, 0)
    probs = np.asarray(model.predict(batch, verbose=0))[0]
    latency_ms = (time.perf_counter() - started) * 1000

    index = int(probs.argmax())
    predicted = config.CLASS_NAMES[index]
    confidence = float(probs[index])

    database.record_prediction(predicted, confidence, latency_ms, version)
    return {
        "class": predicted,
        "confidence": round(confidence, 4),
        "probabilities": {
            name: round(float(p), 4)
            for name, p in zip(config.CLASS_NAMES, probs)
        },
        "latency_ms": round(latency_ms, 2),
        "model_version": version,
    }
