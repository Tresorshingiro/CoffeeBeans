"""Model construction, evaluation, retraining, and the promotion gate."""
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from . import config, database, preprocessing, prediction


def build_model():
    """MobileNetV2 with a 4-way head. Mirrors the notebook exactly.

    Augmentation and preprocess_input live inside the graph so training,
    the notebook, and the API cannot drift in how they preprocess.
    """
    base = MobileNetV2(input_shape=(*config.IMG_SIZE, 3),
                       include_top=False, weights="imagenet")
    base.trainable = False

    inputs = layers.Input(shape=(*config.IMG_SIZE, 3))
    x = preprocessing.build_augmentation()(inputs)
    x = preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(config.CLASS_NAMES), activation="softmax")(x)
    return models.Model(inputs, outputs)


def freeze_batchnorm(model) -> None:
    """Hold every BatchNorm layer frozen, including nested submodels.

    Letting BatchNorm update during fine-tuning is what destabilised the
    first training run in the notebook.
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        if hasattr(layer, "layers"):
            freeze_batchnorm(layer)


def evaluate(model, ds) -> dict:
    """Accuracy, loss, per-class precision/recall/F1, and confusion matrix.

    The dataset must be unshuffled so predictions line up with labels.
    """
    y_true = np.concatenate([y.numpy() for _, y in ds]).astype(int)
    y_prob = np.asarray(model.predict(ds, verbose=0))
    y_pred = y_prob.argmax(axis=1)

    eps = 1e-9
    picked = y_prob[np.arange(len(y_true)), y_true]
    loss = float(-np.mean(np.log(np.clip(picked, eps, 1.0))))

    labels = list(range(len(config.CLASS_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)

    return {
        "accuracy": float((y_true == y_pred).mean()),
        "loss": round(loss, 4),
        "n_samples": int(len(y_true)),
        "per_class": {
            name: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i, name in enumerate(config.CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels).tolist(),
    }


class _ProgressCallback(tf.keras.callbacks.Callback):
    """Streams epoch results to the UI's polling log."""

    def __init__(self, emit):
        super().__init__()
        self.emit = emit

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.emit(
            f"Epoch {epoch + 1}: loss={logs.get('loss', 0):.4f} "
            f"accuracy={logs.get('accuracy', 0):.4f}")


def _gather_training_data(replay_n):
    """Pending images plus a replay slice. Returns (paths, labels, n_replay)."""
    pending_paths, pending_labels = preprocessing.list_images(config.PENDING_DIR)
    if not pending_paths:
        return [], [], 0
    replay_paths, replay_labels = preprocessing.sample_replay(replay_n)
    return (pending_paths + replay_paths,
            pending_labels + replay_labels,
            len(replay_paths))


def _run_finetune(paths, labels, epochs, emit):
    """Load the current champion and continue training from it.

    Resolved through the registry rather than CHAMPION_PATH, so a second
    retrain builds on the first one's promoted model instead of silently
    restarting from the original baseline.
    """
    path, version = prediction.champion_path()
    emit(f"Loading champion {version} as the starting point")
    model = tf.keras.models.load_model(path)
    freeze_batchnorm(model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.RETRAIN_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    ds = preprocessing.dataset_from_paths(paths, labels, shuffle=True)
    model.fit(ds, epochs=epochs, verbose=0,
              callbacks=[_ProgressCallback(emit)])
    return model


def _evaluate_candidate(model):
    paths, labels = preprocessing.eval_slice()
    ds = preprocessing.dataset_from_paths(paths, labels, shuffle=False)
    return evaluate(model, ds)


def push_to_hub(path) -> bool:
    """Persist a promoted model to a HF Model repo, if configured.

    Spaces have an ephemeral filesystem; without this a restart reverts to
    the original champion.
    """
    token = os.environ.get("HF_TOKEN")
    repo = os.environ.get("HF_MODEL_REPO")
    if not token or not repo:
        return False
    try:
        from huggingface_hub import HfApi
        HfApi().upload_file(
            path_or_fileobj=str(path),
            path_in_repo=os.path.basename(str(path)),
            repo_id=repo, repo_type="model", token=token)
        return True
    except Exception:
        return False


def promote(model, metrics: dict) -> str:
    """Save the candidate under a timestamped version and make it champion."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"coffee_model_{stamp}"
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / f"{version}.keras"

    model.save(path)
    database.register_model(version, str(path), metrics["accuracy"])
    prediction.set_model(model, version)
    push_to_hub(path)
    return str(path)


def retrain(progress_cb=None, epochs=None, replay_n=None) -> dict:
    """Fine-tune the champion on staged data, then promote only if it wins."""
    epochs = epochs or config.RETRAIN_EPOCHS
    replay_n = replay_n or config.REPLAY_SAMPLES

    def emit(message):
        if progress_cb:
            progress_cb(message)

    paths, labels, n_replay = _gather_training_data(replay_n)
    if not paths:
        raise ValueError("No pending images to retrain on")
    n_pending = len(paths) - n_replay
    emit(f"Staged {n_pending} new images, replaying {n_replay} from training set")

    model = _run_finetune(paths, labels, epochs, emit)

    emit("Evaluating candidate against the held-out test slice")
    metrics = _evaluate_candidate(model)
    candidate_accuracy = float(metrics["accuracy"])

    champion = database.get_champion()
    champion_accuracy = float(champion["accuracy"]) if champion else 0.0
    emit(f"Candidate {candidate_accuracy:.4f} vs "
         f"champion {champion_accuracy:.4f}")

    if candidate_accuracy > champion_accuracy:
        model_path = promote(model, metrics)
        moved = preprocessing.migrate_pending_to_train()
        emit(f"PROMOTED — {moved} images folded into the training set")
        promoted = True
    else:
        emit("REJECTED — champion kept, staged images retained for a retry")
        model_path = None
        promoted = False

    return {
        "promoted": promoted,
        "candidate_accuracy": candidate_accuracy,
        "champion_accuracy": champion_accuracy,
        "model_path": model_path,
        "metrics": metrics,
        "n_pending": n_pending,
        "n_replay": n_replay,
    }
