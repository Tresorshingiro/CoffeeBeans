import hashlib
import io
import random
import shutil
import zipfile
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

from . import config, database

AUTOTUNE = tf.data.AUTOTUNE


def decode_image_bytes(payload: bytes) -> tf.Tensor:
    """Decode any supported image format and resize to the model's input size.

    Uses decode_image rather than decode_jpeg so PNG uploads work.
    """
    img = tf.io.decode_image(payload, channels=3, expand_animations=False)
    img = tf.image.resize(img, config.IMG_SIZE)
    img.set_shape([*config.IMG_SIZE, 3])
    return img


def list_images(root) -> tuple[list[str], list[int]]:
    """Collect image paths with labels taken from CLASS_NAMES by name.

    Labels are never inferred from directory contents, so a partial set of
    class folders still maps to the correct global indices.
    """
    root = Path(root)
    paths: list[str] = []
    labels: list[int] = []
    for index, class_name in enumerate(config.CLASS_NAMES):
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in config.ALLOWED_IMAGE_SUFFIXES:
                paths.append(str(path))
                labels.append(index)
    return paths, labels


def _read_and_resize(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, config.IMG_SIZE)
    img.set_shape([*config.IMG_SIZE, 3])
    return img, label


def dataset_from_paths(paths, labels, batch=None, shuffle=False):
    batch = batch or config.BATCH
    # Use explicit dtypes to handle empty lists correctly. Without this,
    # tf.data.Dataset.from_tensor_slices defaults to float32 which breaks
    # tf.io.read_file expecting strings.
    ds = tf.data.Dataset.from_tensor_slices((
        tf.constant(paths, dtype=tf.string),
        tf.constant(labels, dtype=tf.int32),
    ))
    if shuffle:
        ds = ds.shuffle(max(len(paths), 1), seed=config.SEED,
                        reshuffle_each_iteration=True)
    ds = ds.map(_read_and_resize, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch).prefetch(AUTOTUNE)


def load_dataset(root, shuffle=False, batch=None):
    paths, labels = list_images(root)
    return dataset_from_paths(paths, labels, batch=batch, shuffle=shuffle)


def build_augmentation():
    """The augmentation block from the notebook, kept identical."""
    return models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augment")


def _class_from_member(name: str):
    """Find the class folder anywhere in the member's path, if present."""
    parts = Path(name).parts
    for part in parts[:-1]:
        if part in config.CLASS_NAMES:
            return part
    return None


def pending_counts() -> dict:
    counts = {}
    for class_name in config.CLASS_NAMES:
        class_dir = config.PENDING_DIR / class_name
        if not class_dir.is_dir():
            continue
        n = sum(1 for p in class_dir.iterdir()
                if p.suffix.lower() in config.ALLOWED_IMAGE_SUFFIXES)
        if n:
            counts[class_name] = n
    return counts


def stage_upload(zip_bytes: bytes, batch_id: str) -> dict:
    """Validate a ZIP of class-foldered images and stage them for retraining."""
    if len(zip_bytes) > config.MAX_UPLOAD_BYTES:
        raise ValueError("Upload exceeds the maximum size of "
                         f"{config.MAX_UPLOAD_BYTES // (1024 * 1024)}MB")
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError("File is not a valid ZIP archive") from exc

    accepted = {name: 0 for name in config.CLASS_NAMES}
    rejected: list[dict] = []
    saw_class_folder = False
    total_decompressed = 0

    for member in archive.infolist():
        if member.is_dir():
            continue
        name = member.filename
        if Path(name).name.startswith("."):
            continue  # __MACOSX resource forks and .DS_Store — silently skipped

        class_name = _class_from_member(name)
        if class_name is None:
            rejected.append({"name": name,
                             "reason": "not inside a recognised class folder"})
            continue
        saw_class_folder = True

        suffix = Path(name).suffix.lower()
        if suffix not in config.ALLOWED_IMAGE_SUFFIXES:
            rejected.append({"name": name, "reason": "unsupported file type"})
            continue

        # Guard against decompression bombs: check file_size before decompressing
        if member.file_size > config.MAX_DECOMPRESSED_BYTES:
            raise ValueError(
                f"Member {name} exceeds maximum decompressed size of "
                f"{config.MAX_DECOMPRESSED_BYTES // (1024 * 1024)}MB")
        if total_decompressed + member.file_size > config.MAX_DECOMPRESSED_BYTES:
            raise ValueError(
                f"Archive exceeds maximum decompressed size of "
                f"{config.MAX_DECOMPRESSED_BYTES // (1024 * 1024)}MB")

        payload = archive.read(member)
        total_decompressed += len(payload)
        digest = hashlib.sha256(payload).hexdigest()
        if database.sha256_exists(digest):
            rejected.append({"name": name, "reason": "duplicate image"})
            continue

        try:
            decode_image_bytes(payload)
        except Exception:
            rejected.append({"name": name, "reason": "could not decode image"})
            continue

        dest_dir = config.PENDING_DIR / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / f"{digest[:16]}{suffix}").write_bytes(payload)
        database.record_upload(batch_id, Path(name).name, class_name, digest)
        accepted[class_name] += 1

    if not saw_class_folder:
        raise ValueError(
            "Archive contains no class folder. Expected top-level folders "
            f"named: {', '.join(config.CLASS_NAMES)}")

    total = sum(accepted.values())
    if total == 0:
        # If all items were rejected as duplicates, that's ok.
        # If they were rejected for other reasons, the archive has no valid images.
        has_only_duplicates = all("duplicate" in r["reason"] for r in rejected)
        if not has_only_duplicates:
            raise ValueError("Archive contained no valid images")

    counts = pending_counts()
    return {
        "batch_id": batch_id,
        "accepted": {k: v for k, v in accepted.items() if v},
        "rejected": rejected,
        "total_accepted": total,
        "pending_counts": counts,
        "pending_total": sum(counts.values()),
    }


def clear_pending() -> None:
    if config.PENDING_DIR.exists():
        shutil.rmtree(config.PENDING_DIR)


def migrate_pending_to_train() -> int:
    """Move staged images into the training set so future retrains replay them."""
    moved = 0
    for class_name in config.CLASS_NAMES:
        src_dir = config.PENDING_DIR / class_name
        if not src_dir.is_dir():
            continue
        dest_dir = config.train_dir() / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for path in list(src_dir.iterdir()):
            if path.suffix.lower() not in config.ALLOWED_IMAGE_SUFFIXES:
                continue
            shutil.move(str(path), str(dest_dir / path.name))
            moved += 1
    clear_pending()
    return moved


def _stratified_sample(root, n: int, seed: int):
    """Draw n images spread evenly across classes, deterministically."""
    per_class = max(1, n // len(config.CLASS_NAMES))
    rng = random.Random(seed)
    paths: list[str] = []
    labels: list[int] = []
    for index, class_name in enumerate(config.CLASS_NAMES):
        class_dir = Path(root) / class_name
        if not class_dir.is_dir():
            continue
        available = sorted(
            str(p) for p in class_dir.iterdir()
            if p.suffix.lower() in config.ALLOWED_IMAGE_SUFFIXES)
        chosen = rng.sample(available, min(per_class, len(available)))
        paths.extend(sorted(chosen))
        labels.extend([index] * len(chosen))
    return paths, labels


def sample_replay(n=None):
    """Sample from the original training set to prevent catastrophic forgetting."""
    return _stratified_sample(config.train_dir(),
                              n or config.REPLAY_SAMPLES, config.SEED)


def eval_slice():
    """Fixed test slice used by the promotion gate, identical across runs."""
    return _stratified_sample(config.test_dir(),
                              config.EVAL_SLICE_SIZE, config.SEED)
