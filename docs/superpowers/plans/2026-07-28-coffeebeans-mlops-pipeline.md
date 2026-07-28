# Coffee Bean MLOps Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deployment half of the coffee bean grading project — `src/` modules, a FastAPI service, a Streamlit UI, a gated retraining pipeline, Docker packaging, and a Locust load test — on top of the existing trained MobileNetV2 model.

**Architecture:** A FastAPI service owns all model work (predict, upload staging, retraining) and persists state to SQLite. A Streamlit UI is a pure HTTP client of that API. nginx fronts both, giving one public URL on Hugging Face Spaces and a scalable API tier locally for the load test. Retraining runs as a threadpool background job with polled status, loads the current champion as its starting point, mixes uploaded data with a replay slice of the original training set, and only promotes the result if it beats the champion on a fixed test slice.

**Tech Stack:** Python 3.12, TensorFlow 2.20 (CPU), FastAPI, Streamlit, SQLite (stdlib `sqlite3`), nginx, Docker + docker-compose 1.29.2, Locust, pytest.

## Global Constraints

- **Python 3.12.3**, TensorFlow pinned to **`tensorflow-cpu==2.20.0`**. The `.keras` champion file was written by TF 2.20 and older versions cannot load it.
- **`CLASS_NAMES = ["defect", "longberry", "peaberry", "premium"]`** — alphabetical, defined once in `src/config.py`. Never let Keras infer labels from directory contents.
- **`docker-compose` v1.29.2 only** (`docker compose` v2 is not installed). Compose file must declare **`version: "2.4"`** — v3's `deploy.resources.limits` is ignored outside swarm mode, so CPU pinning would silently do nothing.
- **Host for load testing: 4 cores, 7GB RAM.** Replica counts are 1, 2, 4. Supabase containers must be stopped before the 4-replica run or it will OOM.
- **Never run `git add .`** until Task 1's `.gitignore` is committed. The working tree holds 124MB of untracked images and model binaries.
- Retrain background job must be a **synchronous `def`**, never `async def`. Starlette dispatches sync background tasks to a threadpool; an async one would block the event loop.
- All API errors return JSON `{"detail": "..."}` with the status codes in Task 9-11. Tracebacks are never surfaced to the UI.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/config.py` | Constants, paths, split resolution (sample vs full dataset) |
| `src/database.py` | SQLite schema and all persistence |
| `src/preprocessing.py` | Image decode, dataset construction, upload staging, replay sampling |
| `src/model.py` | Model build, evaluation, retrain, promotion gate |
| `src/prediction.py` | Champion model lifecycle, single-image inference, hot-swap |
| `api/main.py` | FastAPI routes, job registry, error mapping |
| `ui/app.py` | Streamlit four-page client |
| `scripts/make_sample.py` | One-time split of full dataset into committed sample |
| `scripts/download_data.py` | Fetch full dataset from HF Dataset repo |
| `scripts/build_insights.py` | Precompute visualization payload |
| `locust/locustfile.py` | Load test task mix |
| `docker/nginx.conf` | Local reverse proxy with re-resolving upstream |
| `docker/nginx.space.conf` | Single-port routing for HF Spaces |
| `docker/Dockerfile.space` | All-in-one Space image |
| `Dockerfile` | API-only image, the unit scaled during load test |
| `docker-compose.yml` | Local topology, format 2.4, CPU-pinned replicas |

---

### Task 1: Scaffolding, config, and housekeeping

**Files:**
- Create: `.gitignore`, `requirements.txt`, `requirements-dev.txt`, `src/__init__.py`, `src/config.py`, `tests/__init__.py`, `tests/test_config.py`
- Rename: `models/coffee_model (1).keras` → `models/coffee_model.keras`
- Delete: `data/{train,test}/`

**Interfaces:**
- Consumes: nothing
- Produces: `src/config.py` exposing `CLASS_NAMES: list[str]`, `IMG_SIZE: tuple[int,int]`, `BATCH: int`, `SEED: int`, `RETRAIN_THRESHOLD: int`, `REPLAY_SAMPLES: int`, `EVAL_SLICE_SIZE: int`, `RETRAIN_EPOCHS: int`, `RETRAIN_LR: float`, `MAX_UPLOAD_BYTES: int`, `ALLOWED_IMAGE_SUFFIXES: set[str]`, `PROJECT_ROOT`, `DATA_DIR`, `PENDING_DIR`, `MODELS_DIR`, `DB_PATH`, `CHAMPION_PATH`, `INSIGHTS_PATH: Path`, and functions `train_dir() -> Path`, `test_dir() -> Path`

- [ ] **Step 1: Write `.gitignore` first, before anything else**

```gitignore
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/
.DS_Store

# Full dataset — fetched by scripts/download_data.py
data/full/

# Runtime state
data/pending/
data/app.db
data/app.db-wal
data/app.db-shm
data/insights.json

# Retrained model versions and training checkpoints
models/coffee_model_*.keras
models/best.weights.h5

# Load test output
locust/results/
```

- [ ] **Step 2: Verify the ignore rules actually cover the large files**

```bash
git status --porcelain | wc -l
git status --porcelain | grep -E '^\?\?' | head -20
```

Expected: `data/full/` and `models/best.weights.h5` do NOT appear. `data/train/`, `data/test/`, `models/coffee_model.keras`, `notebook/` DO appear — those are meant to be committed.

Note: `data/train/` and `data/test/` still hold the full 6,400/1,600 images at this point. Do NOT commit them yet — Task 12 replaces them with the sample. Commit only the files listed in Step 7 below.

- [ ] **Step 3: Housekeeping — rename the model, remove the stray directory**

```bash
mv "models/coffee_model (1).keras" models/coffee_model.keras
rmdir "data/{train,test}/premium" "data/{train,test}/defect" \
      "data/{train,test}/longberry" "data/{train,test}/peaberry" 2>/dev/null
rm -rf "data/{train,test}"
ls models/
```

Expected: `models/` contains `best.weights.h5` and `coffee_model.keras`. `data/` contains only `train` and `test`.

- [ ] **Step 4: Write requirements files**

`requirements.txt`:
```
tensorflow-cpu==2.20.0
fastapi==0.115.6
uvicorn[standard]==0.34.0
python-multipart==0.0.20
streamlit==1.41.1
requests==2.32.3
pillow==11.1.0
numpy==2.1.3
scikit-learn==1.6.1
matplotlib==3.10.0
huggingface-hub==0.27.1
```

`requirements-dev.txt`:
```
-r requirements.txt
pytest==8.3.4
httpx==0.28.1
locust==2.32.5
```

- [ ] **Step 5: Create the virtualenv and install**

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -c "import tensorflow as tf; print(tf.__version__)"
```

Expected: prints `2.20.0`. This takes several minutes and downloads ~600MB.

- [ ] **Step 6: Write the failing test**

`tests/test_config.py`:
```python
from src import config


def test_class_names_are_alphabetical():
    # Keras derives labels alphabetically from directory names. If CLASS_NAMES
    # ever diverges from sorted order, every label index silently shifts.
    assert config.CLASS_NAMES == sorted(config.CLASS_NAMES)


def test_class_names_match_train_directories():
    found = sorted(p.name for p in config.train_dir().iterdir() if p.is_dir())
    assert found == config.CLASS_NAMES


def test_train_dir_prefers_full_dataset_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "FULL_DIR", tmp_path / "full")
    (tmp_path / "train").mkdir()
    assert config.train_dir() == tmp_path / "train"

    full_train = tmp_path / "full" / "train"
    full_train.mkdir(parents=True)
    (full_train / "defect").mkdir()
    assert config.train_dir() == full_train
```

- [ ] **Step 7: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 8: Write `src/config.py`**

`src/__init__.py` and `tests/__init__.py` are empty files.

`src/config.py`:
```python
"""Central configuration. Every path and hyperparameter lives here."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FULL_DIR = DATA_DIR / "full"
PENDING_DIR = DATA_DIR / "pending"
MODELS_DIR = PROJECT_ROOT / "models"

DB_PATH = DATA_DIR / "app.db"
INSIGHTS_PATH = DATA_DIR / "insights.json"
CHAMPION_PATH = MODELS_DIR / "coffee_model.keras"

# Alphabetical, and it must stay that way — see tests/test_config.py.
CLASS_NAMES = ["defect", "longberry", "peaberry", "premium"]

IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42

RETRAIN_THRESHOLD = 50      # pending images before retraining unlocks
REPLAY_SAMPLES = 400        # drawn from the training set during retrain
EVAL_SLICE_SIZE = 400       # stratified test images used by the promotion gate
RETRAIN_EPOCHS = 3
RETRAIN_LR = 1e-5

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _resolve_split(name: str) -> Path:
    """Prefer the full dataset when it has been downloaded, else the sample."""
    full = FULL_DIR / name
    if full.is_dir() and any(full.iterdir()):
        return full
    return DATA_DIR / name


def train_dir() -> Path:
    return _resolve_split("train")


def test_dir() -> Path:
    return _resolve_split("test")
```

- [ ] **Step 9: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 10: Commit**

```bash
git add .gitignore requirements.txt requirements-dev.txt \
        src/__init__.py src/config.py tests/__init__.py tests/test_config.py
git commit -m "feat: add gitignore, config module, and project scaffolding"
```

---

### Task 2: SQLite persistence layer

**Files:**
- Create: `src/database.py`, `tests/test_database.py`

**Interfaces:**
- Consumes: `src.config` (`DB_PATH`)
- Produces:
  - `set_db_path(path: Path) -> None`
  - `init_db() -> None`
  - `record_upload(batch_id: str, filename: str, class_label: str, sha256: str) -> None`
  - `sha256_exists(sha256: str) -> bool`
  - `create_retrain_run(n_pending: int, n_replay: int, epochs: int) -> int`
  - `finish_retrain_run(run_id: int, status: str, candidate_accuracy: float | None, champion_accuracy: float | None, promoted: bool, model_path: str | None, log: str) -> None`
  - `get_retrain_run(run_id: int) -> dict | None`
  - `list_retrain_runs(limit: int = 20) -> list[dict]`
  - `record_prediction(predicted_class: str, confidence: float, latency_ms: float, model_version: str) -> None`
  - `prediction_stats() -> dict` with keys `total`, `mean_latency_ms`, `p95_latency_ms`, `class_counts`
  - `register_model(version: str, path: str, accuracy: float) -> None`
  - `get_champion() -> dict | None` with keys `version`, `path`, `accuracy`, `promoted_at`

- [ ] **Step 1: Write the failing test**

`tests/test_database.py`:
```python
import pytest

from src import database


@pytest.fixture(autouse=True)
def temp_db(tmp_path):
    database.set_db_path(tmp_path / "test.db")
    database.init_db()
    yield


def test_record_upload_and_dedupe():
    assert database.sha256_exists("abc123") is False
    database.record_upload("batch-1", "bean.jpg", "defect", "abc123")
    assert database.sha256_exists("abc123") is True


def test_retrain_run_lifecycle():
    run_id = database.create_retrain_run(n_pending=51, n_replay=400, epochs=3)
    running = database.get_retrain_run(run_id)
    assert running["status"] == "running"
    assert running["n_pending"] == 51

    database.finish_retrain_run(
        run_id, status="completed", candidate_accuracy=0.871,
        champion_accuracy=0.865, promoted=True,
        model_path="models/coffee_model_x.keras", log="line one\nline two",
    )
    done = database.get_retrain_run(run_id)
    assert done["status"] == "completed"
    assert done["promoted"] == 1
    assert done["candidate_accuracy"] == pytest.approx(0.871)
    assert done["finished_at"] is not None


def test_list_retrain_runs_newest_first():
    first = database.create_retrain_run(1, 1, 1)
    second = database.create_retrain_run(2, 2, 2)
    runs = database.list_retrain_runs()
    assert [r["id"] for r in runs] == [second, first]


def test_prediction_stats():
    database.record_prediction("defect", 0.9, 100.0, "v1")
    database.record_prediction("defect", 0.8, 200.0, "v1")
    database.record_prediction("premium", 0.7, 300.0, "v1")
    stats = database.prediction_stats()
    assert stats["total"] == 3
    assert stats["mean_latency_ms"] == pytest.approx(200.0)
    assert stats["class_counts"]["defect"] == 2


def test_prediction_stats_empty():
    stats = database.prediction_stats()
    assert stats["total"] == 0
    assert stats["mean_latency_ms"] == 0.0
    assert stats["class_counts"] == {}


def test_register_model_moves_champion():
    database.register_model("v1", "models/a.keras", 0.865)
    assert database.get_champion()["version"] == "v1"
    database.register_model("v2", "models/b.keras", 0.871)
    champ = database.get_champion()
    assert champ["version"] == "v2"
    assert champ["accuracy"] == pytest.approx(0.871)


def test_get_champion_none_when_empty():
    assert database.get_champion() is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_database.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.database'`

- [ ] **Step 3: Write `src/database.py`**

```python
"""SQLite persistence. Stdlib sqlite3, no ORM.

WAL mode is enabled because the retrain background thread writes concurrently
with request handlers.
"""
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path

from . import config

_db_path: Path = config.DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id    TEXT NOT NULL,
    filename    TEXT NOT NULL,
    class_label TEXT NOT NULL,
    sha256      TEXT NOT NULL UNIQUE,
    uploaded_at TEXT NOT NULL,
    used_in_run INTEGER
);

CREATE TABLE IF NOT EXISTS retrain_runs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at         TEXT NOT NULL,
    finished_at        TEXT,
    status             TEXT NOT NULL,
    n_pending          INTEGER NOT NULL,
    n_replay           INTEGER NOT NULL,
    epochs             INTEGER NOT NULL,
    candidate_accuracy REAL,
    champion_accuracy  REAL,
    promoted           INTEGER NOT NULL DEFAULT 0,
    model_path         TEXT,
    log                TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    confidence      REAL NOT NULL,
    latency_ms      REAL NOT NULL,
    model_version   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_registry (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    version     TEXT NOT NULL,
    path        TEXT NOT NULL,
    accuracy    REAL NOT NULL,
    promoted_at TEXT NOT NULL,
    is_champion INTEGER NOT NULL DEFAULT 0
);
"""


def set_db_path(path) -> None:
    global _db_path
    _db_path = Path(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect() -> sqlite3.Connection:
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_db_path, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def record_upload(batch_id, filename, class_label, sha256) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO uploads "
            "(batch_id, filename, class_label, sha256, uploaded_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (batch_id, filename, class_label, sha256, _now()),
        )


def sha256_exists(sha256) -> bool:
    with connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM uploads WHERE sha256 = ?", (sha256,)
        ).fetchone()
    return row is not None


def create_retrain_run(n_pending, n_replay, epochs) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO retrain_runs "
            "(started_at, status, n_pending, n_replay, epochs) "
            "VALUES (?, 'running', ?, ?, ?)",
            (_now(), n_pending, n_replay, epochs),
        )
        return int(cur.lastrowid)


def finish_retrain_run(run_id, status, candidate_accuracy,
                       champion_accuracy, promoted, model_path, log) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE retrain_runs SET finished_at = ?, status = ?, "
            "candidate_accuracy = ?, champion_accuracy = ?, promoted = ?, "
            "model_path = ?, log = ? WHERE id = ?",
            (_now(), status, candidate_accuracy, champion_accuracy,
             int(promoted), model_path, log, run_id),
        )


def get_retrain_run(run_id):
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM retrain_runs WHERE id = ?", (run_id,)
        ).fetchone()
    return dict(row) if row else None


def list_retrain_runs(limit=20):
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM retrain_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def record_prediction(predicted_class, confidence, latency_ms,
                      model_version) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO predictions "
            "(ts, predicted_class, confidence, latency_ms, model_version) "
            "VALUES (?, ?, ?, ?, ?)",
            (_now(), predicted_class, confidence, latency_ms, model_version),
        )


def prediction_stats():
    with connect() as conn:
        latencies = [r[0] for r in conn.execute(
            "SELECT latency_ms FROM predictions").fetchall()]
        counts = conn.execute(
            "SELECT predicted_class, COUNT(*) FROM predictions "
            "GROUP BY predicted_class").fetchall()
    if not latencies:
        return {"total": 0, "mean_latency_ms": 0.0,
                "p95_latency_ms": 0.0, "class_counts": {}}
    ordered = sorted(latencies)
    idx = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "total": len(latencies),
        "mean_latency_ms": round(statistics.mean(latencies), 2),
        "p95_latency_ms": round(ordered[idx], 2),
        "class_counts": {row[0]: row[1] for row in counts},
    }


def register_model(version, path, accuracy) -> None:
    with connect() as conn:
        conn.execute("UPDATE model_registry SET is_champion = 0")
        conn.execute(
            "INSERT INTO model_registry "
            "(version, path, accuracy, promoted_at, is_champion) "
            "VALUES (?, ?, ?, ?, 1)",
            (version, str(path), accuracy, _now()),
        )


def get_champion():
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM model_registry WHERE is_champion = 1 "
            "ORDER BY id DESC LIMIT 1").fetchone()
    return dict(row) if row else None
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_database.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/database.py tests/test_database.py
git commit -m "feat: add SQLite persistence layer"
```

---

### Task 3: Image decoding and dataset construction

**Files:**
- Create: `src/preprocessing.py`, `tests/test_preprocessing.py`, `tests/conftest.py`

**Interfaces:**
- Consumes: `src.config`
- Produces:
  - `decode_image_bytes(payload: bytes) -> tf.Tensor` shape `(224, 224, 3)`, float32
  - `list_images(root: Path) -> tuple[list[str], list[int]]` — label indices from `CLASS_NAMES`
  - `dataset_from_paths(paths: list[str], labels: list[int], batch: int = BATCH, shuffle: bool = False) -> tf.data.Dataset`
  - `load_dataset(root: Path, shuffle: bool = False) -> tf.data.Dataset`
  - `build_augmentation() -> keras.Sequential`

**Design note:** labels are assigned explicitly from `CLASS_NAMES` by `list_images` rather than inferred by `image_dataset_from_directory`. This is the structural fix for the notebook's label-ordering bug — a directory containing only `defect/` and `premium/` cannot produce indices 0 and 1.

- [ ] **Step 1: Write the shared test fixture**

`tests/conftest.py`:
```python
import io

import pytest
from PIL import Image


def _png_bytes(color=(120, 140, 90), size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(color=(120, 140, 90), size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def png_bytes():
    return _png_bytes


@pytest.fixture
def jpeg_bytes():
    return _jpeg_bytes


@pytest.fixture
def image_tree(tmp_path):
    """Builds a directory tree of JPEGs. Returns (root, {class: count})."""
    def _build(root_name, counts):
        root = tmp_path / root_name
        for cls, n in counts.items():
            d = root / cls
            d.mkdir(parents=True)
            for i in range(n):
                (d / f"{i}.jpg").write_bytes(_jpeg_bytes())
        return root
    return _build
```

- [ ] **Step 2: Write the failing test**

`tests/test_preprocessing.py`:
```python
import pytest

from src import config, preprocessing


def test_decode_jpeg(jpeg_bytes):
    img = preprocessing.decode_image_bytes(jpeg_bytes())
    assert tuple(img.shape) == (*config.IMG_SIZE, 3)


def test_decode_png(png_bytes):
    # The notebook used decode_jpeg, which raises on PNG. User uploads
    # will contain PNGs, so decode_image is required here.
    img = preprocessing.decode_image_bytes(png_bytes())
    assert tuple(img.shape) == (*config.IMG_SIZE, 3)


def test_decode_rejects_garbage():
    with pytest.raises(Exception):
        preprocessing.decode_image_bytes(b"this is not an image")


def test_list_images_assigns_labels_from_class_names(image_tree):
    root = image_tree("train", {"defect": 2, "longberry": 3})
    paths, labels = preprocessing.list_images(root)
    assert len(paths) == 5
    assert sorted(set(labels)) == [0, 1]


def test_list_images_partial_classes_keep_global_indices(image_tree):
    # Regression test for the label-ordering bug. Only defect (0) and
    # premium (3) are present; premium must stay 3, not collapse to 1.
    root = image_tree("partial", {"defect": 2, "premium": 2})
    paths, labels = preprocessing.list_images(root)
    assert sorted(set(labels)) == [0, 3]


def test_list_images_ignores_unknown_directories(image_tree):
    root = image_tree("mixed", {"defect": 2})
    (root / "not_a_class").mkdir()
    (root / "not_a_class" / "x.jpg").write_bytes(b"junk")
    paths, labels = preprocessing.list_images(root)
    assert len(paths) == 2


def test_dataset_from_paths_shapes(image_tree):
    root = image_tree("train", {"defect": 4, "premium": 4})
    paths, labels = preprocessing.list_images(root)
    ds = preprocessing.dataset_from_paths(paths, labels, batch=4)
    images, batch_labels = next(iter(ds))
    assert tuple(images.shape) == (4, *config.IMG_SIZE, 3)
    assert tuple(batch_labels.shape) == (4,)


def test_load_dataset_preserves_order_when_not_shuffled(image_tree):
    root = image_tree("test", {"defect": 4, "longberry": 4})
    ds = preprocessing.load_dataset(root, shuffle=False)
    labels = [int(v) for _, batch in ds for v in batch]
    assert labels == sorted(labels)
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_preprocessing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.preprocessing'`

- [ ] **Step 4: Write `src/preprocessing.py`**

```python
"""Image decoding, dataset construction, upload staging, replay sampling."""
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

from . import config

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
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
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
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_preprocessing.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add src/preprocessing.py tests/test_preprocessing.py tests/conftest.py
git commit -m "feat: add image decoding and dataset construction"
```

---

### Task 4: Upload staging

**Files:**
- Modify: `src/preprocessing.py` (append)
- Create: `tests/test_staging.py`

**Interfaces:**
- Consumes: `src.database` (`sha256_exists`, `record_upload`), `decode_image_bytes` from Task 3
- Produces:
  - `stage_upload(zip_bytes: bytes, batch_id: str) -> dict` with keys `batch_id`, `accepted: dict[str,int]`, `rejected: list[dict]`, `total_accepted: int`, `pending_counts: dict[str,int]`, `pending_total: int`. Raises `ValueError` on unusable archives.
  - `pending_counts() -> dict[str, int]`
  - `clear_pending() -> None`
  - `migrate_pending_to_train() -> int`

- [ ] **Step 1: Write the failing test**

`tests/test_staging.py`:
```python
import io
import zipfile

import pytest

from src import config, database, preprocessing


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    database.set_db_path(tmp_path / "test.db")
    database.init_db()
    monkeypatch.setattr(config, "PENDING_DIR", tmp_path / "pending")
    yield


def make_zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, payload in entries.items():
            z.writestr(name, payload)
    return buf.getvalue()


def test_accepts_valid_class_folders(jpeg_bytes):
    blob = make_zip({
        "defect/a.jpg": jpeg_bytes((10, 20, 30)),
        "defect/b.jpg": jpeg_bytes((11, 21, 31)),
        "premium/c.jpg": jpeg_bytes((12, 22, 32)),
    })
    result = preprocessing.stage_upload(blob, "batch-1")
    assert result["accepted"]["defect"] == 2
    assert result["accepted"]["premium"] == 1
    assert result["total_accepted"] == 3
    assert result["pending_total"] == 3


def test_rejects_unknown_class_folder(jpeg_bytes):
    blob = make_zip({
        "defect/a.jpg": jpeg_bytes((10, 20, 30)),
        "arabica/b.jpg": jpeg_bytes((13, 23, 33)),
    })
    result = preprocessing.stage_upload(blob, "batch-2")
    assert result["accepted"]["defect"] == 1
    reasons = {r["name"]: r["reason"] for r in result["rejected"]}
    assert "arabica/b.jpg" in reasons


def test_rejects_unsupported_file_type(jpeg_bytes):
    blob = make_zip({
        "defect/a.jpg": jpeg_bytes(),
        "defect/notes.txt": b"hello",
    })
    result = preprocessing.stage_upload(blob, "batch-3")
    assert result["accepted"]["defect"] == 1
    assert any("unsupported" in r["reason"] for r in result["rejected"])


def test_dedupes_identical_content(jpeg_bytes):
    payload = jpeg_bytes((50, 60, 70))
    blob = make_zip({"defect/a.jpg": payload, "defect/copy.jpg": payload})
    result = preprocessing.stage_upload(blob, "batch-4")
    assert result["total_accepted"] == 1
    assert any("duplicate" in r["reason"] for r in result["rejected"])


def test_dedupes_across_uploads(jpeg_bytes):
    payload = jpeg_bytes((80, 90, 100))
    preprocessing.stage_upload(make_zip({"defect/a.jpg": payload}), "b1")
    second = preprocessing.stage_upload(make_zip({"defect/a.jpg": payload}), "b2")
    assert second["total_accepted"] == 0


def test_ignores_macos_metadata(jpeg_bytes):
    blob = make_zip({
        "defect/a.jpg": jpeg_bytes(),
        "__MACOSX/defect/._a.jpg": b"junk",
    })
    result = preprocessing.stage_upload(blob, "batch-5")
    assert result["total_accepted"] == 1


def test_rejects_corrupt_image():
    blob = make_zip({"defect/broken.jpg": b"not really a jpeg"})
    with pytest.raises(ValueError, match="no valid images"):
        preprocessing.stage_upload(blob, "batch-6")


def test_rejects_non_zip():
    with pytest.raises(ValueError, match="valid ZIP"):
        preprocessing.stage_upload(b"definitely not a zip", "batch-7")


def test_rejects_archive_with_no_class_folders(jpeg_bytes):
    blob = make_zip({"loose.jpg": jpeg_bytes()})
    with pytest.raises(ValueError, match="class folder"):
        preprocessing.stage_upload(blob, "batch-8")


def test_rejects_oversized_upload():
    with pytest.raises(ValueError, match="maximum size"):
        preprocessing.stage_upload(
            b"x" * (config.MAX_UPLOAD_BYTES + 1), "batch-9")


def test_migrate_pending_to_train(jpeg_bytes, tmp_path, monkeypatch):
    train = tmp_path / "train"
    monkeypatch.setattr(config, "train_dir", lambda: train)
    blob = make_zip({"defect/a.jpg": jpeg_bytes((5, 5, 5))})
    preprocessing.stage_upload(blob, "batch-10")

    moved = preprocessing.migrate_pending_to_train()
    assert moved == 1
    assert len(list((train / "defect").glob("*.jpg"))) == 1
    assert preprocessing.pending_counts() == {}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_staging.py -v`
Expected: FAIL with `AttributeError: module 'src.preprocessing' has no attribute 'stage_upload'`

- [ ] **Step 3: Append the staging code to `src/preprocessing.py`**

Add these imports at the top of the file: `import hashlib`, `import io`, `import shutil`, `import zipfile`, and `from . import database`.

```python
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

        payload = archive.read(member)
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_staging.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing.py tests/test_staging.py
git commit -m "feat: add ZIP upload staging with validation and dedupe"
```

---

### Task 5: Replay sampling and evaluation slice

**Files:**
- Modify: `src/preprocessing.py` (append)
- Create: `tests/test_sampling.py`

**Interfaces:**
- Consumes: `list_images` from Task 3
- Produces:
  - `sample_replay(n: int) -> tuple[list[str], list[int]]`
  - `eval_slice() -> tuple[list[str], list[int]]` — fixed, seeded, stratified, size `EVAL_SLICE_SIZE`

- [ ] **Step 1: Write the failing test**

`tests/test_sampling.py`:
```python
from src import config, preprocessing


def test_sample_replay_is_stratified(image_tree, monkeypatch):
    root = image_tree("train", {c: 50 for c in config.CLASS_NAMES})
    monkeypatch.setattr(config, "train_dir", lambda: root)
    paths, labels = preprocessing.sample_replay(40)
    assert len(paths) == 40
    for index in range(len(config.CLASS_NAMES)):
        assert labels.count(index) == 10


def test_sample_replay_handles_small_classes(image_tree, monkeypatch):
    root = image_tree("train", {"defect": 3, "longberry": 50,
                                "peaberry": 50, "premium": 50})
    monkeypatch.setattr(config, "train_dir", lambda: root)
    paths, labels = preprocessing.sample_replay(40)
    assert labels.count(0) == 3      # capped by what exists
    assert labels.count(1) == 10


def test_sample_replay_is_deterministic(image_tree, monkeypatch):
    root = image_tree("train", {c: 50 for c in config.CLASS_NAMES})
    monkeypatch.setattr(config, "train_dir", lambda: root)
    first, _ = preprocessing.sample_replay(20)
    second, _ = preprocessing.sample_replay(20)
    assert first == second


def test_eval_slice_is_fixed_and_stratified(image_tree, monkeypatch):
    root = image_tree("test", {c: 200 for c in config.CLASS_NAMES})
    monkeypatch.setattr(config, "test_dir", lambda: root)
    monkeypatch.setattr(config, "EVAL_SLICE_SIZE", 40)
    first, labels = preprocessing.eval_slice()
    second, _ = preprocessing.eval_slice()
    assert first == second               # same slice every run
    assert len(first) == 40
    for index in range(len(config.CLASS_NAMES)):
        assert labels.count(index) == 10
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_sampling.py -v`
Expected: FAIL with `AttributeError: module 'src.preprocessing' has no attribute 'sample_replay'`

- [ ] **Step 3: Append to `src/preprocessing.py`**

Add `import random` to the imports.

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_sampling.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing.py tests/test_sampling.py
git commit -m "feat: add stratified replay sampling and fixed eval slice"
```

---

### Task 6: Prediction service and model lifecycle

**Files:**
- Create: `src/prediction.py`, `tests/test_prediction.py`

**Interfaces:**
- Consumes: `src.config`, `src.database`, `src.preprocessing.decode_image_bytes`
- Produces:
  - `load_champion() -> None` — called once at startup
  - `set_model(model, version: str) -> None` — atomic hot-swap
  - `get_model()` — returns the served model or `None`
  - `get_version() -> str`
  - `is_ready() -> bool`
  - `predict_image(payload: bytes) -> dict` with keys `class`, `confidence`, `probabilities`, `latency_ms`, `model_version`
  - `warmup() -> None`

- [ ] **Step 1: Write the failing test**

`tests/test_prediction.py`:
```python
import numpy as np
import pytest

from src import config, database, prediction


class StubModel:
    """Returns a fixed probability vector without loading TensorFlow weights."""

    def __init__(self, probs):
        self.probs = np.array([probs], dtype="float32")
        self.calls = 0

    def predict(self, x, verbose=0):
        self.calls += 1
        return self.probs


@pytest.fixture(autouse=True)
def isolated(tmp_path):
    database.set_db_path(tmp_path / "test.db")
    database.init_db()
    prediction.set_model(None, "none")
    yield


def test_not_ready_without_model():
    assert prediction.is_ready() is False


def test_predict_returns_highest_class(jpeg_bytes):
    prediction.set_model(StubModel([0.1, 0.7, 0.1, 0.1]), "v1")
    result = prediction.predict_image(jpeg_bytes())
    assert result["class"] == "longberry"
    assert result["confidence"] == pytest.approx(0.7, abs=1e-4)
    assert result["model_version"] == "v1"
    assert result["latency_ms"] > 0


def test_predict_returns_all_probabilities(jpeg_bytes):
    prediction.set_model(StubModel([0.1, 0.7, 0.1, 0.1]), "v1")
    result = prediction.predict_image(jpeg_bytes())
    assert set(result["probabilities"]) == set(config.CLASS_NAMES)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)


def test_predict_records_to_database(jpeg_bytes):
    prediction.set_model(StubModel([0.1, 0.7, 0.1, 0.1]), "v1")
    prediction.predict_image(jpeg_bytes())
    assert database.prediction_stats()["total"] == 1


def test_predict_raises_when_no_model(jpeg_bytes):
    with pytest.raises(RuntimeError, match="not loaded"):
        prediction.predict_image(jpeg_bytes())


def test_hot_swap_changes_served_model(jpeg_bytes):
    prediction.set_model(StubModel([0.9, 0.1, 0.0, 0.0]), "v1")
    assert prediction.predict_image(jpeg_bytes())["class"] == "defect"
    prediction.set_model(StubModel([0.0, 0.0, 0.0, 1.0]), "v2")
    result = prediction.predict_image(jpeg_bytes())
    assert result["class"] == "premium"
    assert result["model_version"] == "v2"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_prediction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.prediction'`

- [ ] **Step 3: Write `src/prediction.py`**

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_prediction.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/prediction.py tests/test_prediction.py
git commit -m "feat: add prediction service with atomic model hot-swap"
```

---

### Task 7: Model construction and evaluation

**Files:**
- Create: `src/model.py`, `tests/test_model_eval.py`

**Interfaces:**
- Consumes: `src.preprocessing` (`build_augmentation`, `load_dataset`)
- Produces:
  - `build_model() -> keras.Model`
  - `evaluate(model, ds) -> dict` with keys `accuracy`, `loss`, `per_class` (dict of class → `{precision, recall, f1, support}`), `confusion_matrix` (list of lists), `n_samples`
  - `freeze_batchnorm(model) -> None`

- [ ] **Step 1: Write the failing test**

`tests/test_model_eval.py`:
```python
import numpy as np
import pytest
import tensorflow as tf

from src import config, model as model_module


class PerfectStub:
    """Predicts each sample's true label with full confidence."""

    def __init__(self, labels):
        self.labels = labels

    def predict(self, ds, verbose=0):
        out = np.zeros((len(self.labels), len(config.CLASS_NAMES)),
                       dtype="float32")
        for row, label in enumerate(self.labels):
            out[row, label] = 1.0
        return out


def make_ds(labels):
    images = tf.zeros((len(labels), *config.IMG_SIZE, 3))
    return tf.data.Dataset.from_tensor_slices(
        (images, np.array(labels, dtype="int32"))).batch(2)


def test_evaluate_perfect_predictions():
    labels = [0, 1, 2, 3, 0, 1, 2, 3]
    result = model_module.evaluate(PerfectStub(labels), make_ds(labels))
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["n_samples"] == 8
    assert result["per_class"]["defect"]["recall"] == pytest.approx(1.0)


def test_evaluate_reports_all_four_metrics():
    labels = [0, 1, 2, 3]
    result = model_module.evaluate(PerfectStub(labels), make_ds(labels))
    assert set(result) >= {"accuracy", "loss", "per_class", "confusion_matrix"}
    for class_name in config.CLASS_NAMES:
        assert set(result["per_class"][class_name]) == {
            "precision", "recall", "f1", "support"}


def test_evaluate_confusion_matrix_shape():
    labels = [0, 1, 2, 3]
    result = model_module.evaluate(PerfectStub(labels), make_ds(labels))
    matrix = result["confusion_matrix"]
    assert len(matrix) == 4
    assert all(len(row) == 4 for row in matrix)


def test_evaluate_imperfect_predictions():
    true_labels = [0, 0, 1, 1]
    predicted = [0, 1, 1, 1]
    result = model_module.evaluate(PerfectStub(predicted), make_ds(true_labels))
    assert result["accuracy"] == pytest.approx(0.75)


def test_freeze_batchnorm():
    built = model_module.build_model()
    model_module.freeze_batchnorm(built)
    bn_layers = [layer for layer in built.get_layer("mobilenetv2_1.00_224").layers
                 if isinstance(layer, tf.keras.layers.BatchNormalization)]
    assert bn_layers
    assert all(layer.trainable is False for layer in bn_layers)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_model_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.model'`

- [ ] **Step 3: Write `src/model.py`**

```python
"""Model construction, evaluation, retraining, and the promotion gate."""
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from . import config, preprocessing


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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_model_eval.py -v`
Expected: 5 passed. The `freeze_batchnorm` test downloads ImageNet weights on first run (~9MB).

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model_eval.py
git commit -m "feat: add model construction and evaluation metrics"
```

---

### Task 8: Retraining with the promotion gate

**Files:**
- Modify: `src/model.py` (append)
- Create: `tests/test_retrain.py`

**Interfaces:**
- Consumes: everything from Tasks 2-7
- Produces:
  - `retrain(progress_cb=None, epochs: int | None = None, replay_n: int | None = None) -> dict` with keys `promoted`, `candidate_accuracy`, `champion_accuracy`, `model_path`, `metrics`, `n_pending`, `n_replay`
  - `promote(model, metrics: dict) -> str` — returns the saved path
  - `push_to_hub(path: str) -> bool`

- [ ] **Step 1: Write the failing test**

`tests/test_retrain.py`:
```python
import pytest

from src import config, database, model as model_module, prediction


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    database.set_db_path(tmp_path / "test.db")
    database.init_db()
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    (tmp_path / "models").mkdir()
    prediction.set_model(None, "none")
    yield


class FakeModel:
    def __init__(self, accuracy):
        self.accuracy = accuracy
        self.saved_to = None

    def save(self, path):
        self.saved_to = str(path)
        open(path, "w").close()


def test_promote_saves_versioned_file_and_registers():
    fake = FakeModel(0.9)
    path = model_module.promote(fake, {"accuracy": 0.871})
    assert "coffee_model_" in path
    assert path.endswith(".keras")
    champion = database.get_champion()
    assert champion["accuracy"] == pytest.approx(0.871)
    assert champion["path"] == path


def test_promote_hot_swaps_served_model():
    fake = FakeModel(0.9)
    model_module.promote(fake, {"accuracy": 0.9})
    assert prediction.is_ready() is True
    assert prediction.get_version().startswith("coffee_model_")


def test_gate_promotes_when_candidate_wins(monkeypatch):
    monkeypatch.setattr(model_module, "_run_finetune",
                        lambda *a, **k: FakeModel(0.9))
    monkeypatch.setattr(model_module, "_evaluate_candidate",
                        lambda model: {"accuracy": 0.9})
    monkeypatch.setattr(model_module, "_gather_training_data",
                        lambda replay_n: (["p"] * 51, [0] * 51, 400))
    database.register_model("v0", "models/old.keras", 0.865)

    result = model_module.retrain()
    assert result["promoted"] is True
    assert result["candidate_accuracy"] == pytest.approx(0.9)


def test_gate_rejects_when_candidate_loses(monkeypatch):
    monkeypatch.setattr(model_module, "_run_finetune",
                        lambda *a, **k: FakeModel(0.5))
    monkeypatch.setattr(model_module, "_evaluate_candidate",
                        lambda model: {"accuracy": 0.5})
    monkeypatch.setattr(model_module, "_gather_training_data",
                        lambda replay_n: (["p"] * 51, [0] * 51, 400))
    database.register_model("v0", "models/old.keras", 0.865)

    result = model_module.retrain()
    assert result["promoted"] is False
    assert result["model_path"] is None
    # champion untouched
    assert database.get_champion()["version"] == "v0"


def test_retrain_raises_without_pending_data(monkeypatch):
    monkeypatch.setattr(model_module, "_gather_training_data",
                        lambda replay_n: ([], [], 0))
    with pytest.raises(ValueError, match="No pending images"):
        model_module.retrain()


def test_progress_callback_receives_lines(monkeypatch):
    monkeypatch.setattr(model_module, "_run_finetune",
                        lambda *a, **k: FakeModel(0.9))
    monkeypatch.setattr(model_module, "_evaluate_candidate",
                        lambda model: {"accuracy": 0.9})
    monkeypatch.setattr(model_module, "_gather_training_data",
                        lambda replay_n: (["p"] * 51, [0] * 51, 400))
    database.register_model("v0", "models/old.keras", 0.865)

    lines = []
    model_module.retrain(progress_cb=lines.append)
    assert any("Staged" in line for line in lines)
    assert any("PROMOTED" in line for line in lines)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_retrain.py -v`
Expected: FAIL with `AttributeError: module 'src.model' has no attribute 'promote'`

- [ ] **Step 3: Append the retrain code to `src/model.py`**

Add imports: `import os`, `from datetime import datetime`, `from . import database, prediction`.

```python
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
    """Load the champion and continue training from it."""
    model = tf.keras.models.load_model(config.CHAMPION_PATH)
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

    emit("Loading current champion as the starting point")
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_retrain.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: all passing (about 50 tests)

- [ ] **Step 6: Commit**

```bash
git add src/model.py tests/test_retrain.py
git commit -m "feat: add retraining with replay and promotion gate"
```

---

### Task 9: API core — health, status, predict

**Files:**
- Create: `api/__init__.py`, `api/main.py`, `tests/test_api_core.py`

**Interfaces:**
- Consumes: `src.prediction`, `src.database`, `src.preprocessing`, `src.config`
- Produces: FastAPI `app` with `/api/health`, `/api/status`, `/api/predict`; module-level `START_TIME: float`

- [ ] **Step 1: Write the failing test**

`tests/test_api_core.py`:
```python
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient


class StubModel:
    def predict(self, x, verbose=0):
        return np.array([[0.05, 0.85, 0.05, 0.05]], dtype="float32")


@pytest.fixture
def client(tmp_path, monkeypatch):
    from src import config, database, prediction
    database.set_db_path(tmp_path / "test.db")
    database.init_db()
    monkeypatch.setattr(config, "PENDING_DIR", tmp_path / "pending")

    from api import main
    prediction.set_model(StubModel(), "test-v1")
    return TestClient(main.app)


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_status_reports_uptime_and_model(client):
    body = client.get("/api/status").json()
    assert body["model_loaded"] is True
    assert body["model_version"] == "test-v1"
    assert body["uptime_seconds"] >= 0
    assert body["pending_total"] == 0
    assert body["retrain_ready"] is False
    assert body["retrain_threshold"] == 50


def test_predict_returns_class_and_confidence(client, jpeg_bytes):
    response = client.post(
        "/api/predict",
        files={"file": ("bean.jpg", io.BytesIO(jpeg_bytes()), "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["class"] == "longberry"
    assert body["confidence"] == pytest.approx(0.85, abs=1e-3)
    assert len(body["probabilities"]) == 4


def test_predict_rejects_undecodable_file(client):
    response = client.post(
        "/api/predict",
        files={"file": ("bad.jpg", io.BytesIO(b"nope"), "image/jpeg")})
    assert response.status_code == 415


def test_predict_503_when_model_missing(client):
    from src import prediction
    prediction.set_model(None, "none")
    response = client.post(
        "/api/predict",
        files={"file": ("x.jpg", io.BytesIO(b"abc"), "image/jpeg")})
    assert response.status_code == 503


def test_status_counts_predictions(client, jpeg_bytes):
    client.post("/api/predict",
                files={"file": ("a.jpg", io.BytesIO(jpeg_bytes()), "image/jpeg")})
    body = client.get("/api/status").json()
    assert body["predictions_served"] == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_api_core.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api'`

- [ ] **Step 3: Write `api/main.py`**

`api/__init__.py` is an empty file.

```python
"""FastAPI service. All model work happens here; the UI is a pure client."""
import time

from fastapi import FastAPI, File, HTTPException, UploadFile

from src import config, database, prediction, preprocessing

START_TIME = time.time()

app = FastAPI(
    title="Coffee Bean Grading API",
    description="Grades green Arabica coffee beans into four classes.",
    version="1.0.0",
)


@app.on_event("startup")
def startup() -> None:
    database.init_db()
    try:
        prediction.load_champion()
        prediction.warmup()
    except Exception as exc:  # noqa: BLE001 — service must start to report this
        print(f"WARNING: could not load champion model: {exc}")


@app.get("/api/health")
def health():
    """Liveness only. Deliberately does not touch the model."""
    return {"status": "ok"}


@app.get("/api/status")
def status():
    counts = preprocessing.pending_counts()
    pending_total = sum(counts.values())
    stats = database.prediction_stats()
    champion = database.get_champion()
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "model_loaded": prediction.is_ready(),
        "model_version": prediction.get_version(),
        "model_accuracy": champion["accuracy"] if champion else None,
        "classes": config.CLASS_NAMES,
        "pending_counts": counts,
        "pending_total": pending_total,
        "retrain_threshold": config.RETRAIN_THRESHOLD,
        "retrain_ready": pending_total >= config.RETRAIN_THRESHOLD,
        "predictions_served": stats["total"],
        "mean_latency_ms": stats["mean_latency_ms"],
        "p95_latency_ms": stats["p95_latency_ms"],
        "class_counts": stats["class_counts"],
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not prediction.is_ready():
        raise HTTPException(503, "Model is not loaded")
    payload = await file.read()
    try:
        return prediction.predict_image(payload)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(415, "Could not decode image") from exc
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_api_core.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add api/__init__.py api/main.py tests/test_api_core.py
git commit -m "feat: add API health, status, and predict endpoints"
```

---

### Task 10: API upload and retrain endpoints

**Files:**
- Modify: `api/main.py` (append)
- Create: `tests/test_api_retrain.py`

**Interfaces:**
- Consumes: `src.model.retrain`, `src.preprocessing.stage_upload`
- Produces: `/api/upload`, `/api/retrain`, `/api/retrain/{job_id}`, `/api/retrain/history`; module-level `_JOBS: dict[int, dict]` and `_retrain_lock: threading.Lock`

- [ ] **Step 1: Write the failing test**

`tests/test_api_retrain.py`:
```python
import io
import zipfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    from src import config, database
    database.set_db_path(tmp_path / "test.db")
    database.init_db()
    monkeypatch.setattr(config, "PENDING_DIR", tmp_path / "pending")

    from api import main
    main._JOBS.clear()
    return TestClient(main.app)


def make_zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, payload in entries.items():
            z.writestr(name, payload)
    return buf.getvalue()


def test_upload_accepts_valid_zip(client, jpeg_bytes):
    blob = make_zip({"defect/a.jpg": jpeg_bytes((1, 2, 3)),
                     "premium/b.jpg": jpeg_bytes((4, 5, 6))})
    response = client.post(
        "/api/upload",
        files={"file": ("beans.zip", io.BytesIO(blob), "application/zip")})
    assert response.status_code == 200
    body = response.json()
    assert body["total_accepted"] == 2
    assert body["pending_total"] == 2
    assert body["retrain_ready"] is False


def test_upload_rejects_bad_zip(client):
    response = client.post(
        "/api/upload",
        files={"file": ("x.zip", io.BytesIO(b"not a zip"), "application/zip")})
    assert response.status_code == 400
    assert "ZIP" in response.json()["detail"]


def test_upload_rejects_archive_without_class_folders(client, jpeg_bytes):
    blob = make_zip({"loose.jpg": jpeg_bytes()})
    response = client.post(
        "/api/upload",
        files={"file": ("x.zip", io.BytesIO(blob), "application/zip")})
    assert response.status_code == 400


def test_retrain_422_below_threshold(client, jpeg_bytes):
    blob = make_zip({"defect/a.jpg": jpeg_bytes()})
    client.post("/api/upload",
                files={"file": ("b.zip", io.BytesIO(blob), "application/zip")})
    response = client.post("/api/retrain", json={})
    assert response.status_code == 422
    assert "threshold" in response.json()["detail"].lower()


def test_retrain_force_bypasses_threshold(client, jpeg_bytes, monkeypatch):
    from src import model as model_module
    monkeypatch.setattr(model_module, "retrain", lambda **kw: {
        "promoted": True, "candidate_accuracy": 0.9, "champion_accuracy": 0.86,
        "model_path": "models/x.keras", "metrics": {"accuracy": 0.9},
        "n_pending": 1, "n_replay": 400})

    blob = make_zip({"defect/a.jpg": jpeg_bytes()})
    client.post("/api/upload",
                files={"file": ("b.zip", io.BytesIO(blob), "application/zip")})
    response = client.post("/api/retrain", json={"force": True})
    assert response.status_code == 202
    assert "job_id" in response.json()


def test_retrain_status_reports_completion(client, jpeg_bytes, monkeypatch):
    from src import model as model_module
    monkeypatch.setattr(model_module, "retrain", lambda **kw: {
        "promoted": True, "candidate_accuracy": 0.9, "champion_accuracy": 0.86,
        "model_path": "models/x.keras", "metrics": {"accuracy": 0.9},
        "n_pending": 1, "n_replay": 400})

    blob = make_zip({"defect/a.jpg": jpeg_bytes()})
    client.post("/api/upload",
                files={"file": ("b.zip", io.BytesIO(blob), "application/zip")})
    job_id = client.post("/api/retrain", json={"force": True}).json()["job_id"]

    body = client.get(f"/api/retrain/{job_id}").json()
    assert body["status"] == "completed"
    assert body["promoted"] is True


def test_retrain_status_404_for_unknown_job(client):
    assert client.get("/api/retrain/9999").status_code == 404


def test_retrain_409_when_already_running(client, jpeg_bytes, monkeypatch):
    from api import main
    main._retrain_lock.acquire()
    try:
        blob = make_zip({"defect/a.jpg": jpeg_bytes()})
        client.post("/api/upload",
                    files={"file": ("b.zip", io.BytesIO(blob), "application/zip")})
        response = client.post("/api/retrain", json={"force": True})
        assert response.status_code == 409
    finally:
        main._retrain_lock.release()


def test_retrain_history_is_listed(client):
    assert client.get("/api/retrain/history").status_code == 200
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_api_retrain.py -v`
Expected: FAIL with 404 on `/api/upload`

- [ ] **Step 3: Append to `api/main.py`**

Add imports: `import threading`, `import uuid`, `from fastapi import BackgroundTasks`, `from pydantic import BaseModel`, `from src import database, model as model_module`.

```python
_JOBS: dict = {}
_retrain_lock = threading.Lock()


class RetrainRequest(BaseModel):
    epochs: int | None = None
    replay_n: int | None = None
    force: bool = False


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    payload = await file.read()
    if len(payload) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Upload exceeds the maximum size")
    try:
        result = preprocessing.stage_upload(payload, str(uuid.uuid4())[:8])
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    result["retrain_threshold"] = config.RETRAIN_THRESHOLD
    result["retrain_ready"] = result["pending_total"] >= config.RETRAIN_THRESHOLD
    return result


def _run_retrain_job(run_id: int, epochs, replay_n) -> None:
    """Synchronous by design.

    Starlette dispatches sync background tasks to a threadpool. An async def
    here would block the event loop and freeze the status polling that draws
    the progress log.
    """
    job = _JOBS[run_id]
    try:
        result = model_module.retrain(
            progress_cb=job["log"].append, epochs=epochs, replay_n=replay_n)
        job.update(status="completed", promoted=result["promoted"],
                   metrics=result["metrics"],
                   candidate_accuracy=result["candidate_accuracy"],
                   champion_accuracy=result["champion_accuracy"])
        database.finish_retrain_run(
            run_id, "completed", result["candidate_accuracy"],
            result["champion_accuracy"], result["promoted"],
            result["model_path"], "\n".join(job["log"]))
    except Exception as exc:  # noqa: BLE001
        job.update(status="failed", error=str(exc))
        job["log"].append(f"FAILED: {exc}")
        database.finish_retrain_run(
            run_id, "failed", None, None, False, None, "\n".join(job["log"]))
    finally:
        if _retrain_lock.locked():
            _retrain_lock.release()


@app.post("/api/retrain", status_code=202)
def trigger_retrain(request: RetrainRequest, background: BackgroundTasks):
    counts = preprocessing.pending_counts()
    pending_total = sum(counts.values())
    if pending_total == 0:
        raise HTTPException(422, "No pending images to retrain on")
    if pending_total < config.RETRAIN_THRESHOLD and not request.force:
        raise HTTPException(
            422,
            f"Only {pending_total} pending images; threshold is "
            f"{config.RETRAIN_THRESHOLD}. Upload more or use force.")
    if not _retrain_lock.acquire(blocking=False):
        raise HTTPException(409, "A retraining run is already in progress")

    epochs = request.epochs or config.RETRAIN_EPOCHS
    replay_n = request.replay_n or config.REPLAY_SAMPLES
    run_id = database.create_retrain_run(pending_total, replay_n, epochs)
    _JOBS[run_id] = {"status": "running", "log": [], "promoted": None,
                     "metrics": None, "error": None}
    background.add_task(_run_retrain_job, run_id, epochs, replay_n)
    return {"job_id": run_id, "status": "running",
            "n_pending": pending_total, "n_replay": replay_n}


@app.get("/api/retrain/history")
def retrain_history():
    return {"runs": database.list_retrain_runs()}


@app.get("/api/retrain/{job_id}")
def retrain_status(job_id: int):
    job = _JOBS.get(job_id)
    if job is None:
        record = database.get_retrain_run(job_id)
        if record is None:
            raise HTTPException(404, f"No retraining job with id {job_id}")
        return {"job_id": job_id, "status": record["status"],
                "log": (record["log"] or "").splitlines(),
                "promoted": bool(record["promoted"]),
                "candidate_accuracy": record["candidate_accuracy"],
                "champion_accuracy": record["champion_accuracy"]}
    return {"job_id": job_id, **job}
```

**Route ordering matters:** `/api/retrain/history` must be declared before `/api/retrain/{job_id}`, or FastAPI will try to parse `"history"` as an integer and return 422.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_api_retrain.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add api/main.py tests/test_api_retrain.py
git commit -m "feat: add upload and background retraining endpoints"
```

---

### Task 11: Insights and production metrics

**Files:**
- Create: `scripts/build_insights.py`, `tests/test_api_insights.py`
- Modify: `api/main.py` (append)

**Interfaces:**
- Consumes: `src.preprocessing`, `src.model.evaluate`
- Produces:
  - `scripts/build_insights.py` writing `config.INSIGHTS_PATH` with keys `class_counts`, `channel_means`, `area_ratios`, `interpretations`, `generated_at`
  - `/api/insights` and `/api/metrics`

- [ ] **Step 1: Write the failing test**

`tests/test_api_insights.py`:
```python
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient


class StubModel:
    def predict(self, ds, verbose=0):
        n = sum(int(batch.shape[0]) for _, batch in ds)
        out = np.zeros((n, 4), dtype="float32")
        out[:, 0] = 1.0
        return out


@pytest.fixture
def client(tmp_path, monkeypatch):
    from src import config, database
    database.set_db_path(tmp_path / "test.db")
    database.init_db()
    monkeypatch.setattr(config, "INSIGHTS_PATH", tmp_path / "insights.json")
    monkeypatch.setattr(config, "PENDING_DIR", tmp_path / "pending")
    from api import main
    return TestClient(main.app)


def test_insights_404_when_not_generated(client):
    assert client.get("/api/insights").status_code == 404


def test_insights_served_when_present(client, tmp_path):
    from src import config
    payload = {"class_counts": {"defect": 1600}, "interpretations": {}}
    config.INSIGHTS_PATH.write_text(json.dumps(payload))
    body = client.get("/api/insights").json()
    assert body["class_counts"]["defect"] == 1600


def test_metrics_returns_four_metric_families(client, image_tree, monkeypatch):
    from src import config, prediction
    root = image_tree("test", {c: 2 for c in config.CLASS_NAMES})
    monkeypatch.setattr(config, "test_dir", lambda: root)
    prediction.set_model(StubModel(), "test-v1")

    body = client.get("/api/metrics").json()
    assert set(body) >= {"accuracy", "loss", "per_class", "confusion_matrix"}
    assert body["n_samples"] == 8


def test_metrics_503_without_model(client):
    from src import prediction
    prediction.set_model(None, "none")
    assert client.get("/api/metrics").status_code == 503
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_api_insights.py -v`
Expected: FAIL with 404 on `/api/metrics`

- [ ] **Step 3: Append endpoints to `api/main.py`**

Add imports: `import json`, `from fastapi.responses import JSONResponse`.

```python
@app.get("/api/insights")
def insights():
    if not config.INSIGHTS_PATH.exists():
        raise HTTPException(
            404, "Insights not generated. Run scripts/build_insights.py")
    return JSONResponse(json.loads(config.INSIGHTS_PATH.read_text()))


@app.get("/api/metrics")
def metrics():
    """Evaluate the deployed champion against the full test set.

    This is the production-evaluation surface: its numbers should line up
    with the notebook's.
    """
    if not prediction.is_ready():
        raise HTTPException(503, "Model is not loaded")
    ds = preprocessing.load_dataset(config.test_dir(), shuffle=False)
    result = model_module.evaluate(prediction.get_model(), ds)
    result["model_version"] = prediction.get_version()
    return result
```

- [ ] **Step 4: Write `scripts/build_insights.py`**

```python
"""Precompute the dataset visualizations served by /api/insights.

Recomputing these per page load on a 2-vCPU Space would be unusably slow.
Run once after the dataset is in place:  python scripts/build_insights.py
"""
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

SAMPLE_PER_CLASS = 150


def _class_files(class_name):
    class_dir = config.train_dir() / class_name
    return sorted(p for p in class_dir.iterdir()
                  if p.suffix.lower() in config.ALLOWED_IMAGE_SUFFIXES)


def class_counts():
    return {c: len(_class_files(c)) for c in config.CLASS_NAMES}


def channel_means(rng):
    out = {}
    for class_name in config.CLASS_NAMES:
        files = _class_files(class_name)
        chosen = rng.sample(files, min(SAMPLE_PER_CLASS, len(files)))
        stacked = np.stack([
            np.asarray(Image.open(f).convert("RGB").resize((64, 64)),
                       dtype="float32")
            for f in chosen])
        means = stacked.reshape(-1, 3).mean(axis=0)
        out[class_name] = {"r": round(float(means[0]), 2),
                           "g": round(float(means[1]), 2),
                           "b": round(float(means[2]), 2)}
    return out


def area_ratios(rng):
    """Fraction of pixels belonging to the bean rather than the background.

    Beans are photographed on a plain white background, so a simple
    brightness threshold segments them well enough to compare silhouettes.
    """
    out = {}
    for class_name in config.CLASS_NAMES:
        files = _class_files(class_name)
        chosen = rng.sample(files, min(SAMPLE_PER_CLASS, len(files)))
        ratios = []
        for path in chosen:
            gray = np.asarray(
                Image.open(path).convert("L").resize((128, 128)),
                dtype="float32")
            ratios.append(float((gray < 200).mean()))
        out[class_name] = {
            "mean": round(float(np.mean(ratios)), 4),
            "std": round(float(np.std(ratios)), 4),
            "p25": round(float(np.percentile(ratios, 25)), 4),
            "p75": round(float(np.percentile(ratios, 75)), 4),
        }
    return out


INTERPRETATIONS = {
    "class_balance":
        "The four classes are exactly balanced at 1,600 training images each. "
        "That means plain accuracy is an honest headline metric and no "
        "resampling or class weighting is needed — a skewed dataset would "
        "have forced both.",
    "channel_means":
        "Green leads in every class, as expected for green coffee beans, but "
        "defect runs measurably darker across all three channels. Black and "
        "insect-damaged beans drag its average down, which means colour alone "
        "carries real discriminative signal for that class.",
    "area_ratios":
        "Bean area separates the classes by shape rather than colour. "
        "Longberry occupies a distinctly elongated footprint and peaberry a "
        "compact one. Defect shows the widest spread of any class — defective "
        "beans vary enormously in size and form.",
    "story":
        "The exploratory analysis predicted the model's weakness before "
        "training ran. Defect is the most visually varied class on both shape "
        "and colour, so it was always going to be the hardest to pin down — "
        "and it came back with the lowest recall in the confusion matrix "
        "(0.767 against 0.930 for longberry). The data told us where the "
        "model would struggle, and it did.",
}


def main():
    rng = random.Random(config.SEED)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "class_counts": class_counts(),
        "channel_means": channel_means(rng),
        "area_ratios": area_ratios(rng),
        "interpretations": INTERPRETATIONS,
    }
    config.INSIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.INSIGHTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"wrote {config.INSIGHTS_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_api_insights.py -v`
Expected: 4 passed

- [ ] **Step 6: Generate the real insights file and check it**

```bash
.venv/bin/python scripts/build_insights.py
.venv/bin/python -c "import json; d=json.load(open('data/insights.json')); print(d['class_counts']); print(d['channel_means']['defect']); print(d['area_ratios']['longberry'])"
```

Expected: counts of 1600 per class, and defect's RGB means visibly lower than the other classes'. If defect is *not* darker, the interpretation text in `INTERPRETATIONS` must be corrected to match the actual data rather than left as written.

- [ ] **Step 7: Commit**

```bash
git add api/main.py scripts/build_insights.py tests/test_api_insights.py
git commit -m "feat: add dataset insights and production metrics endpoints"
```

---

### Task 12: Dataset sample and download scripts

**Files:**
- Create: `scripts/make_sample.py`, `scripts/download_data.py`

**Interfaces:**
- Consumes: `src.config`
- Produces: `data/full/{train,test}/` holding the full dataset, `data/{train,test}/` holding the committed sample

**Note:** this task moves 82MB of images on disk. Run the migration once and verify counts before committing.

- [ ] **Step 1: Write `scripts/make_sample.py`**

```python
"""Move the full dataset to data/full/ and cut a committable sample.

Run once. The full dataset is gitignored; the sample is what lives in Git so
a fresh clone is small but still functional.

    python scripts/make_sample.py
"""
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

TRAIN_PER_CLASS = 200
TEST_PER_CLASS = 100


def migrate_to_full():
    """Move data/train and data/test under data/full, once."""
    config.FULL_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        source = config.DATA_DIR / split
        target = config.FULL_DIR / split
        if target.exists():
            print(f"{target} already exists, skipping migration")
            continue
        if not source.exists():
            print(f"{source} missing, nothing to migrate")
            continue
        shutil.move(str(source), str(target))
        print(f"moved {source} -> {target}")


def cut_sample(split, per_class):
    rng = random.Random(config.SEED)
    source_root = config.FULL_DIR / split
    target_root = config.DATA_DIR / split
    for class_name in config.CLASS_NAMES:
        source = source_root / class_name
        target = target_root / class_name
        target.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in source.iterdir()
                       if p.suffix.lower() in config.ALLOWED_IMAGE_SUFFIXES)
        for path in rng.sample(files, min(per_class, len(files))):
            shutil.copy2(path, target / path.name)
        print(f"{split}/{class_name}: {len(list(target.iterdir()))} images")


if __name__ == "__main__":
    migrate_to_full()
    cut_sample("train", TRAIN_PER_CLASS)
    cut_sample("test", TEST_PER_CLASS)
```

- [ ] **Step 2: Run it and verify the counts**

```bash
.venv/bin/python scripts/make_sample.py
for d in data/train/* data/test/*; do echo "$d: $(ls $d | wc -l)"; done
du -sh data/train data/test data/full
```

Expected: 200 per class under `data/train`, 100 per class under `data/test`, roughly 12MB combined, and `data/full` holding the original 82MB.

- [ ] **Step 3: Confirm the sample is what Git sees**

```bash
git status --porcelain data/ | head -5
git check-ignore -v data/full/train/defect 2>/dev/null || echo "NOT IGNORED — fix .gitignore"
```

Expected: `data/full` is reported as ignored. If it is not, stop and fix `.gitignore` before continuing.

- [ ] **Step 4: Write `scripts/download_data.py`**

```python
"""Fetch the full USK-Coffee dataset from a Hugging Face Dataset repo.

Set HF_DATA_REPO to your dataset repo id, e.g. "yourname/usk-coffee".
The repo should contain train/<class>/ and test/<class>/ folders.

    python scripts/download_data.py

If this fails, the service still runs on the committed sample — the
promotion gate just scores against fewer images.
"""
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

DEFAULT_REPO = os.environ.get("HF_DATA_REPO", "")


def main() -> int:
    if not DEFAULT_REPO:
        print("HF_DATA_REPO is not set — skipping download, "
              "the committed sample will be used")
        return 0
    if config.FULL_DIR.exists() and any(config.FULL_DIR.iterdir()):
        print(f"{config.FULL_DIR} already populated, nothing to do")
        return 0
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed — skipping download")
        return 0

    try:
        path = snapshot_download(repo_id=DEFAULT_REPO, repo_type="dataset")
    except Exception as exc:  # noqa: BLE001
        print(f"download failed ({exc}) — falling back to the sample")
        return 0

    config.FULL_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        source = Path(path) / split
        if source.is_dir():
            shutil.copytree(source, config.FULL_DIR / split,
                            dirs_exist_ok=True)
            print(f"populated {config.FULL_DIR / split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Verify the full test suite still passes after the data move**

Run: `.venv/bin/pytest tests/ -v`
Expected: all passing. `tests/test_config.py::test_class_names_match_train_directories` now resolves through `data/full/train`.

- [ ] **Step 6: Commit the scripts and the sample**

```bash
git add scripts/make_sample.py scripts/download_data.py data/train data/test
git commit -m "feat: add dataset sample and download scripts"
git count-objects -vH | grep size-pack
```

Expected: pack size in the low tens of MB, not hundreds.

---

### Task 13: Streamlit UI

**Files:**
- Create: `ui/app.py`

**Interfaces:**
- Consumes: the API over HTTP at `API_BASE` (env var `API_BASE`, default `http://localhost:8000`)
- Produces: a four-page Streamlit app

- [ ] **Step 1: Write `ui/app.py`**

```python
"""Streamlit UI. A pure HTTP client of the API — no model code lives here."""
import os
import time

import pandas as pd
import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Coffee Bean Grading", page_icon="☕",
                   layout="wide")


def api_get(path, **kwargs):
    return requests.get(f"{API_BASE}{path}", timeout=120, **kwargs)


def api_post(path, **kwargs):
    return requests.post(f"{API_BASE}{path}", timeout=600, **kwargs)


def show_error(response):
    try:
        detail = response.json().get("detail", response.text)
    except Exception:
        detail = response.text
    st.error(f"{response.status_code}: {detail}")


st.sidebar.title("☕ Coffee Bean Grading")
page = st.sidebar.radio(
    "Page", ["Predict", "Insights", "Data & Retrain", "Monitoring"])

try:
    status = api_get("/api/status").json()
    st.sidebar.success(f"API up · {status['uptime_seconds']:.0f}s")
    st.sidebar.caption(f"Model: {status['model_version']}")
except Exception:
    status = None
    st.sidebar.error("API unreachable")


# ── Predict ──────────────────────────────────────────────────────────────
if page == "Predict":
    st.title("Predict a bean grade")
    st.caption("Upload a single close-up photo of one green coffee bean on a "
               "plain background.")

    uploaded = st.file_uploader("Bean image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        left, right = st.columns([1, 1])
        with left:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)
        if st.button("Predict", type="primary"):
            response = api_post(
                "/api/predict",
                files={"file": (uploaded.name, uploaded.getvalue(),
                                uploaded.type)})
            if response.status_code != 200:
                show_error(response)
            else:
                body = response.json()
                with right:
                    st.metric("Predicted grade", body["class"],
                              f"{body['confidence'] * 100:.1f}% confidence")
                    st.caption(f"Served by {body['model_version']} in "
                               f"{body['latency_ms']:.0f} ms")
                    st.bar_chart(pd.Series(body["probabilities"],
                                           name="probability"))


# ── Insights ─────────────────────────────────────────────────────────────
elif page == "Insights":
    st.title("What the data says")
    response = api_get("/api/insights")
    if response.status_code != 200:
        show_error(response)
        st.info("Run `python scripts/build_insights.py` to generate these.")
    else:
        data = response.json()
        notes = data["interpretations"]

        st.subheader("1. Class balance")
        st.bar_chart(pd.Series(data["class_counts"], name="training images"))
        st.info(notes["class_balance"])

        st.subheader("2. Average colour per class")
        channels = pd.DataFrame(data["channel_means"]).T
        channels.columns = ["Red", "Green", "Blue"]
        st.bar_chart(channels)
        st.info(notes["channel_means"])

        st.subheader("3. Bean area — how much of the frame the bean fills")
        areas = pd.DataFrame(data["area_ratios"]).T
        st.bar_chart(areas[["mean"]])
        st.dataframe(areas, use_container_width=True)
        st.info(notes["area_ratios"])

        st.subheader("The story these tell together")
        st.success(notes["story"])


# ── Data & Retrain ───────────────────────────────────────────────────────
elif page == "Data & Retrain":
    st.title("Upload data and retrain")

    st.subheader("1. Upload new labelled beans")
    st.caption("A .zip containing folders named defect, longberry, peaberry, "
               "or premium.")
    archive = st.file_uploader("ZIP archive", type=["zip"])
    if archive is not None and st.button("Upload", type="primary"):
        response = api_post(
            "/api/upload",
            files={"file": (archive.name, archive.getvalue(),
                            "application/zip")})
        if response.status_code != 200:
            show_error(response)
        else:
            body = response.json()
            st.success(f"Staged {body['total_accepted']} images")
            st.dataframe(
                pd.DataFrame(
                    [{"class": k, "accepted": v}
                     for k, v in body["accepted"].items()]),
                use_container_width=True)
            if body["rejected"]:
                with st.expander(f"{len(body['rejected'])} files rejected"):
                    st.dataframe(pd.DataFrame(body["rejected"]),
                                 use_container_width=True)

    st.divider()
    st.subheader("2. Retrain")
    if status:
        pending = status["pending_total"]
        threshold = status["retrain_threshold"]
        st.progress(min(pending / threshold, 1.0),
                    text=f"{pending} of {threshold} images staged")
        if status["pending_counts"]:
            st.dataframe(
                pd.DataFrame([{"class": k, "pending": v}
                              for k, v in status["pending_counts"].items()]),
                use_container_width=True)

        ready = status["retrain_ready"]
        if ready:
            st.success("Threshold reached — retraining is unlocked.")
        else:
            st.warning(f"{threshold - pending} more images needed to unlock "
                       "retraining automatically.")
        force = st.checkbox("Force retrain below threshold", value=not ready)

        if st.button("Start retraining", type="primary",
                     disabled=pending == 0):
            response = api_post("/api/retrain", json={"force": force})
            if response.status_code != 202:
                show_error(response)
            else:
                job_id = response.json()["job_id"]
                st.session_state["job_id"] = job_id

    job_id = st.session_state.get("job_id")
    if job_id:
        st.caption(f"Job {job_id}")
        log_box = st.empty()
        status_box = st.empty()
        for _ in range(400):
            job = api_get(f"/api/retrain/{job_id}").json()
            log_box.code("\n".join(job.get("log", [])) or "starting…")
            if job["status"] != "running":
                break
            status_box.info("Retraining in progress…")
            time.sleep(2)
        else:
            job = api_get(f"/api/retrain/{job_id}").json()

        if job["status"] == "completed":
            if job.get("promoted"):
                status_box.success(
                    f"Promoted — new model beat the champion "
                    f"({job['candidate_accuracy']:.4f} vs "
                    f"{job['champion_accuracy']:.4f})")
            else:
                status_box.warning(
                    f"Rejected — candidate did not beat the champion "
                    f"({job['candidate_accuracy']:.4f} vs "
                    f"{job['champion_accuracy']:.4f}). Champion kept.")
        elif job["status"] == "failed":
            status_box.error(f"Retraining failed: {job.get('error')}")

    st.divider()
    st.subheader("3. Retraining history")
    history = api_get("/api/retrain/history")
    if history.status_code == 200:
        runs = history.json()["runs"]
        if runs:
            frame = pd.DataFrame(runs)[
                ["id", "started_at", "status", "n_pending", "n_replay",
                 "candidate_accuracy", "champion_accuracy", "promoted"]]
            st.dataframe(frame, use_container_width=True)
            scored = frame.dropna(subset=["candidate_accuracy"])
            if not scored.empty:
                st.line_chart(
                    scored.set_index("id")[["candidate_accuracy",
                                            "champion_accuracy"]])
        else:
            st.caption("No retraining runs yet.")


# ── Monitoring ───────────────────────────────────────────────────────────
elif page == "Monitoring":
    st.title("Service and model monitoring")
    if not status:
        st.error("API unreachable.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Uptime", f"{status['uptime_seconds'] / 60:.1f} min")
        c2.metric("Model", status["model_version"])
        c3.metric("Predictions served", status["predictions_served"])
        c4.metric("Mean latency", f"{status['mean_latency_ms']:.0f} ms")

        if status["class_counts"]:
            st.subheader("What the model has been predicting")
            st.bar_chart(pd.Series(status["class_counts"], name="predictions"))

        st.subheader("Evaluation in production")
        st.caption("The deployed model scored against the held-out test set.")
        if st.button("Run evaluation"):
            with st.spinner("Evaluating…"):
                response = api_get("/api/metrics")
            if response.status_code != 200:
                show_error(response)
            else:
                body = response.json()
                a, b, c = st.columns(3)
                a.metric("Accuracy", f"{body['accuracy']:.3f}")
                b.metric("Loss", f"{body['loss']:.3f}")
                c.metric("Samples", body["n_samples"])
                st.dataframe(pd.DataFrame(body["per_class"]).T,
                             use_container_width=True)
                st.subheader("Confusion matrix")
                st.dataframe(
                    pd.DataFrame(body["confusion_matrix"],
                                 index=[f"actual {c}" for c in status["classes"]],
                                 columns=[f"pred {c}" for c in status["classes"]]),
                    use_container_width=True)
```

- [ ] **Step 2: Run the API and UI together and click through every page**

```bash
.venv/bin/uvicorn api.main:app --port 8000 &
.venv/bin/streamlit run ui/app.py --server.port 8501
```

Verify by hand: Predict returns a class for a real test image; Insights renders three charts with interpretation text; Data & Retrain accepts a ZIP and shows staged counts; Monitoring shows uptime and runs an evaluation.

- [ ] **Step 3: Build a test ZIP and exercise the full retrain loop**

```bash
mkdir -p /tmp/beans/longberry /tmp/beans/peaberry
cp $(ls data/test/longberry/*.jpg | head -30) /tmp/beans/longberry/
cp $(ls data/test/peaberry/*.jpg | head -30) /tmp/beans/peaberry/
(cd /tmp/beans && zip -r /tmp/new_beans.zip .)
```

Upload `/tmp/new_beans.zip` through the UI, confirm 60 images stage and the threshold unlocks, then start retraining and watch the log stream to a promote-or-reject verdict.

- [ ] **Step 4: Commit**

```bash
git add ui/app.py
git commit -m "feat: add Streamlit UI with predict, insights, retrain, monitoring"
```

---

### Task 14: Docker packaging and local compose

**Files:**
- Create: `Dockerfile`, `docker/nginx.conf`, `docker-compose.yml`, `.dockerignore`

**Interfaces:**
- Consumes: `requirements.txt`, `api/`, `ui/`, `src/`, `models/`, `data/`
- Produces: an API image, and a compose topology where `api` scales

- [ ] **Step 1: Write `.dockerignore`**

```
.git/
.venv/
.pytest_cache/
__pycache__/
data/full/
data/pending/
data/app.db*
docs/
notebook/
locust/results/
tests/
```

- [ ] **Step 2: Write `Dockerfile` (API only — this is the unit that scales)**

```dockerfile
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY data/ ./data/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1"]
```

Single worker is deliberate: replica count is the independent variable in the load test, so each container must be exactly one unit of capacity.

- [ ] **Step 3: Write `docker/nginx.conf`**

```nginx
events { worker_connections 1024; }

http {
    access_log off;

    server {
        listen 80;
        client_max_body_size 250M;

        # Docker's embedded DNS. Without re-resolving through a variable,
        # nginx caches one replica's IP at startup and every request lands
        # on the same container no matter what --scale says.
        resolver 127.0.0.11 valid=10s ipv6=off;

        location /api/ {
            set $api_upstream http://api:8000;
            proxy_pass $api_upstream;
            proxy_set_header Host $host;
            proxy_read_timeout 900s;
        }

        location /docs {
            set $docs_upstream http://api:8000;
            proxy_pass $docs_upstream;
            proxy_set_header Host $host;
        }

        location /openapi.json {
            set $openapi_upstream http://api:8000;
            proxy_pass $openapi_upstream;
            proxy_set_header Host $host;
        }

        location / {
            set $ui_upstream http://ui:8501;
            proxy_pass $ui_upstream;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_read_timeout 900s;
        }
    }
}
```

- [ ] **Step 4: Write `docker-compose.yml`**

```yaml
# Format 2.4 is required: docker-compose 1.29.2 ignores v3's
# deploy.resources.limits outside swarm, so CPU pinning would do nothing.
version: "2.4"

services:
  api:
    build: .
    cpus: 1.0
    mem_limit: 1500m
    environment:
      TF_NUM_INTRAOP_THREADS: "1"
      TF_NUM_INTEROP_THREADS: "1"
    volumes:
      - ./data:/app/data
      - ./models:/app/models

  ui:
    build: .
    command: streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0
    environment:
      API_BASE: http://api:8000
    volumes:
      - ./ui:/app/ui
    depends_on:
      - api

  nginx:
    image: nginx:1.27-alpine
    ports:
      - "8080:80"
    volumes:
      - ./docker/nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - api
      - ui
```

The `ui` service reuses the API image and overrides the command, so `streamlit` and `requests` must be in `requirements.txt` — they are. Add `COPY ui/ ./ui/` to the Dockerfile before building.

- [ ] **Step 5: Add the UI copy line to the Dockerfile**

Insert after `COPY api/ ./api/`:
```dockerfile
COPY ui/ ./ui/
```

- [ ] **Step 6: Build and verify a single replica works**

```bash
docker-compose build
docker-compose up -d
sleep 45
curl -s localhost:8080/api/health
curl -s localhost:8080/api/status | head -c 400
```

Expected: `{"status":"ok"}` then a status body with `"model_loaded":true`.

- [ ] **Step 7: Verify scaling actually distributes requests**

```bash
docker-compose up -d --scale api=3
sleep 40
for i in $(seq 1 12); do curl -s localhost:8080/api/status | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['uptime_seconds'])"; done
```

Expected: at least two distinct uptime values, proving requests reach different containers. If every value is identical, the nginx `resolver` block is not working — fix it before running the load test.

- [ ] **Step 8: Commit**

```bash
git add Dockerfile .dockerignore docker/nginx.conf docker-compose.yml
git commit -m "feat: add Docker packaging and scalable compose topology"
```

---

### Task 15: Locust load test

**Files:**
- Create: `locust/locustfile.py`, `locust/run_load_tests.sh`

**Interfaces:**
- Consumes: a running compose stack at `localhost:8080`
- Produces: CSV results per replica count under `locust/results/`

- [ ] **Step 1: Write `locust/locustfile.py`**

```python
"""Load profile: mostly predictions, some status polls.

Run against the nginx front door so replica count is what varies:
    locust -f locust/locustfile.py --host http://localhost:8080
"""
import random
from pathlib import Path

from locust import HttpUser, between, task

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_POOL: list[bytes] = []


def _load_pool(limit=40):
    """Read a few test images into memory once, so disk IO isn't measured."""
    pool = []
    for class_dir in sorted((PROJECT_ROOT / "data" / "test").iterdir()):
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.glob("*.jpg"))[:limit // 4]:
            pool.append(path.read_bytes())
    return pool


class CoffeeBeanUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self):
        global IMAGE_POOL
        if not IMAGE_POOL:
            IMAGE_POOL = _load_pool()

    @task(9)
    def predict(self):
        payload = random.choice(IMAGE_POOL)
        self.client.post(
            "/api/predict",
            files={"file": ("bean.jpg", payload, "image/jpeg")},
            name="/api/predict")

    @task(1)
    def status(self):
        self.client.get("/api/status", name="/api/status")
```

- [ ] **Step 2: Write `locust/run_load_tests.sh`**

```bash
#!/usr/bin/env bash
# Runs an identical load profile at 1, 2, and 4 API replicas.
#
# Host here is 4 cores / 7GB. Stop other containers first or the
# 4-replica run will be memory starved:
#     docker stop $(docker ps -q --filter name=supabase)
set -euo pipefail

USERS=50
SPAWN_RATE=5
DURATION=2m
HOST=http://localhost:8080

mkdir -p locust/results

for N in 1 2 4; do
    echo "=== Scaling API to $N replica(s) ==="
    docker-compose up -d --scale api="$N"
    echo "Waiting for containers to warm up..."
    sleep 60

    until curl -sf "$HOST/api/health" > /dev/null; do sleep 2; done

    echo "=== Running load test with $N replica(s) ==="
    locust -f locust/locustfile.py \
        --host "$HOST" \
        --users "$USERS" \
        --spawn-rate "$SPAWN_RATE" \
        --run-time "$DURATION" \
        --headless \
        --csv "locust/results/replicas_${N}" \
        --csv-full-history

    echo "=== Done: $N replica(s) ==="
    sleep 10
done

echo
echo "Summary:"
for N in 1 2 4; do
    echo "--- $N replica(s) ---"
    column -s, -t < "locust/results/replicas_${N}_stats.csv" | head -3
done
```

- [ ] **Step 3: Free memory, then run the suite**

```bash
docker stop $(docker ps -q --filter name=supabase) || true
free -g
chmod +x locust/run_load_tests.sh
./locust/run_load_tests.sh 2>&1 | tee locust/results/run.log
```

Expected: three completed runs, six CSVs in `locust/results/`. Watch for failures in the 4-replica run — if the failure rate spikes, check `docker stats` for memory pressure and note it in the README rather than hiding it.

- [ ] **Step 4: Extract the numbers for the README**

```bash
for N in 1 2 4; do
  echo "--- $N replicas ---"
  python3 -c "
import csv, sys
with open('locust/results/replicas_${N}_stats.csv') as f:
    for row in csv.DictReader(f):
        if row['Name'] == 'Aggregated':
            print('RPS', row['Requests/s'], '| median', row['Median Response Time'],
                  '| p95', row['95%'], '| p99', row['99%'],
                  '| failures', row['Failure Count'])
"
done
```

Record these in the README table in Task 17.

- [ ] **Step 5: Commit**

```bash
git add locust/locustfile.py locust/run_load_tests.sh
git commit -m "feat: add Locust load test and replica scaling harness"
```

---

### Task 16: Hugging Face Space packaging

**Files:**
- Create: `docker/Dockerfile.space`, `docker/nginx.space.conf`, `docker/start.sh`, `README_HF.md`

**Interfaces:**
- Consumes: the whole application
- Produces: an image serving UI, API, and Swagger on port 7860

- [ ] **Step 1: Write `docker/nginx.space.conf`**

```nginx
events { worker_connections 1024; }

http {
    access_log off;

    server {
        listen 7860;
        client_max_body_size 250M;

        location /api/ {
            proxy_pass http://127.0.0.1:8000;
            proxy_set_header Host $host;
            proxy_read_timeout 900s;
        }

        location /docs {
            proxy_pass http://127.0.0.1:8000;
            proxy_set_header Host $host;
        }

        location /openapi.json {
            proxy_pass http://127.0.0.1:8000;
            proxy_set_header Host $host;
        }

        location / {
            proxy_pass http://127.0.0.1:8501;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_read_timeout 900s;
        }
    }
}
```

A single upstream address is fine here — there is only ever one API process on a Space, so the re-resolving trick from Task 14 is unnecessary.

- [ ] **Step 2: Write `docker/start.sh`**

```bash
#!/usr/bin/env bash
set -e

python scripts/download_data.py || echo "dataset download skipped"
python scripts/build_insights.py || echo "insights generation skipped"

uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1 &
API_PID=$!

streamlit run ui/app.py \
    --server.port 8501 \
    --server.address 127.0.0.1 \
    --server.headless true \
    --browser.gatherUsageStats false &
UI_PID=$!

nginx -c /app/docker/nginx.space.conf -g 'daemon off;' &
NGINX_PID=$!

# If any process dies, take the container down so the Space restarts it.
wait -n $API_PID $UI_PID $NGINX_PID
exit $?
```

- [ ] **Step 3: Write `docker/Dockerfile.space`**

```dockerfile
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    API_BASE=http://127.0.0.1:8000 \
    HOME=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        nginx libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY ui/ ./ui/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY data/ ./data/
COPY docker/nginx.space.conf ./docker/nginx.space.conf
COPY docker/start.sh ./docker/start.sh

RUN chmod +x docker/start.sh \
    && mkdir -p /app/data/pending /var/lib/nginx /var/log/nginx \
    && chmod -R 777 /app/data /var/lib/nginx /var/log/nginx

EXPOSE 7860
CMD ["./docker/start.sh"]
```

- [ ] **Step 4: Test the Space image locally before pushing**

```bash
docker build -f docker/Dockerfile.space -t coffee-space .
docker run --rm -p 7860:7860 coffee-space &
sleep 60
curl -s localhost:7860/api/health
curl -s -o /dev/null -w "%{http_code}\n" localhost:7860/docs
curl -s -o /dev/null -w "%{http_code}\n" localhost:7860/
```

Expected: `{"status":"ok"}`, then `200` twice. Fix any failure here before pushing — debugging a Space through its build log is far slower.

- [ ] **Step 5: Write `README_HF.md` (the Space's own README with required frontmatter)**

```markdown
---
title: Coffee Bean Grading
emoji: ☕
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Coffee Bean Grading

Grades green Arabica coffee beans into defect, longberry, peaberry, or premium.

- UI: this page
- API docs: `/docs`
- Health: `/api/health`

See the GitHub repository for full documentation.
```

- [ ] **Step 6: Deploy to a Space**

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create coffee-bean-grading --type space --space_sdk docker

git remote add space https://huggingface.co/spaces/<your-username>/coffee-bean-grading
cp docker/Dockerfile.space Dockerfile.space
git add Dockerfile.space README_HF.md
git commit -m "chore: add Space deployment files"
git push space main
```

The Space needs `Dockerfile` at the repo root or a `dockerfile_path`. Simplest route: on the Space branch only, copy `docker/Dockerfile.space` to `Dockerfile`. Set `HF_TOKEN` and `HF_MODEL_REPO` as Space secrets so promoted models persist across restarts.

- [ ] **Step 7: Verify the live Space**

```bash
SPACE=https://<your-username>-coffee-bean-grading.hf.space
curl -s $SPACE/api/health
curl -s $SPACE/api/status | head -c 300
```

Expected: healthy, `"model_loaded": true`. Then open the Space and run one prediction through the UI.

- [ ] **Step 8: Commit**

```bash
git add docker/Dockerfile.space docker/nginx.space.conf docker/start.sh README_HF.md
git commit -m "feat: add Hugging Face Space packaging"
```

---

### Task 17: README and final verification

**Files:**
- Create: `README.md`
- Modify: `notebook/coffee_bean_classification.ipynb` (fix the `retrain` label bug)

**Interfaces:**
- Consumes: results from Tasks 15 and 16
- Produces: the submission-facing documentation

- [ ] **Step 1: Fix the notebook's retrain cell so it matches the shipped pipeline**

In the cell defining `retrain` (cell index 35), change the dataset call to pass class names explicitly, and note the replay strategy:

```python
    new_ds = tf.keras.utils.image_dataset_from_directory(
        new_data_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        class_names=class_names,   # never let Keras infer these — a partial
                                   # upload would silently remap every label
    ).prefetch(AUTOTUNE)
```

Add a markdown note below it: the deployed pipeline additionally mixes in a replay sample from the original training set and gates promotion on test accuracy. See `src/model.py`.

- [ ] **Step 2: Write `README.md`**

Fill the bracketed values with the real numbers from Tasks 15 and 16. Do not leave any bracket unfilled.

````markdown
# Coffee Bean Grading — MLOps Pipeline

Grades green Arabica coffee beans into four quality classes from a single
photograph: **defect**, **longberry**, **peaberry**, **premium**.

Coffee is one of Rwanda's largest exports and grading is still done by eye at
most cooperatives. This is an end-to-end pipeline — training, serving,
monitoring, and retraining on new data — around a model that does it
automatically.

- **Live app:** [URL]
- **API docs:** [URL]/docs
- **Video demo:** [YouTube URL]

## Model

MobileNetV2 transfer learning on the USK-Coffee dataset (Febriana et al.,
2022): 8,000 images, 256×256, balanced across four classes.

Training ran in two phases — a frozen ImageNet base, then fine-tuning the top
30 layers at 1e-5 with BatchNorm held frozen. Optimizations: transfer learning,
dropout regularization, early stopping on validation loss, `ReduceLROnPlateau`
scheduling, and checkpointing that only overwrites on improvement.

### Test set results (1,600 held-out images)

| Metric | Value |
|---|---|
| Accuracy | 0.865 |
| Loss | 0.402 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| defect | 0.906 | 0.767 | 0.831 | 400 |
| longberry | 0.905 | 0.930 | 0.917 | 400 |
| peaberry | 0.869 | 0.895 | 0.882 | 400 |
| premium | 0.792 | 0.868 | 0.828 | 400 |

## What the data says

**1. Class balance.** Exactly 1,600 training images per class, so plain accuracy
is an honest headline metric and no resampling was needed.

**2. Colour.** Green dominates every class, but defect runs measurably darker in
all three channels — black and insect-damaged beans pull its average down.
Colour alone carries real signal for that class.

**3. Bean area.** Longberry occupies an elongated footprint, peaberry a compact
one, and defect shows by far the widest spread of any class.

**The story:** the exploratory analysis predicted the model's weakness before
training ran. Defect is the most visually varied class on both shape and
colour, and it came back with the lowest recall (0.767 against longberry's
0.930). The data showed where the model would struggle, and it did.

## Architecture

```
Browser ──► nginx ──┬──► Streamlit UI (:8501)
                    └──► FastAPI (:8000) ──► TensorFlow model
                                         └──► SQLite
Locust  ──► nginx ──► FastAPI replicas
```

## Retraining pipeline

1. **Upload** — a ZIP with class-named folders. Each file is validated,
   deduped by SHA-256, staged to `data/pending/<class>/`, and recorded in SQLite.
2. **Trigger** — retraining unlocks automatically once 50 images are staged;
   a force option overrides the threshold manually.
3. **Preprocess** — staged images are decoded, resized to 224×224, and mixed
   with a 400-image stratified replay sample from the original training set so
   the model cannot forget classes absent from the upload.
4. **Retrain** — the **current champion `.keras` model is loaded as the
   starting point**, BatchNorm stays frozen, and it fine-tunes for 3 epochs at
   1e-5.
5. **Gate** — the candidate is scored on a fixed held-out test slice and is
   only promoted if it beats the champion. On promotion the staged images join
   the training set and the live model hot-swaps with no restart.

## Setup

### Local

```bash
git clone <repo-url> && cd CoffeeBeans
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt

# Optional: fetch the full dataset (the repo ships a sample)
HF_DATA_REPO=<your-dataset-repo> .venv/bin/python scripts/download_data.py

.venv/bin/python scripts/build_insights.py
.venv/bin/pytest tests/ -v

.venv/bin/uvicorn api.main:app --port 8000 &
.venv/bin/streamlit run ui/app.py --server.port 8501
```

UI at http://localhost:8501, API docs at http://localhost:8000/docs.

### Docker

```bash
docker-compose up --build
```

Everything is served through http://localhost:8080.

### Cloud (Hugging Face Spaces)

See `README_HF.md`. Set `HF_TOKEN` and `HF_MODEL_REPO` as Space secrets so
retrained models survive restarts.

## Flood request simulation

Locust, identical profile per run: **50 users, spawn rate 5/s, 2 minutes**,
90% `POST /api/predict` and 10% `GET /api/status`, all through nginx.

**Host:** [CPU model], 4 cores, 7GB RAM, Docker [version].

Each replica is pinned to **1.0 CPU** with `TF_NUM_INTRAOP_THREADS=1`. Without
this, TensorFlow claims every core in a single container, one replica saturates
the host, and adding more shows no gain — the pinning is what makes replica
count a genuine independent variable.

| Containers | RPS | Median (ms) | p95 (ms) | p99 (ms) | Failures |
|---|---|---|---|---|---|
| 1 | [ ] | [ ] | [ ] | [ ] | [ ] |
| 2 | [ ] | [ ] | [ ] | [ ] | [ ] |
| 4 | [ ] | [ ] | [ ] | [ ] | [ ] |

**Production (live Space, [N] users):** [RPS] RPS, [median] ms median.

**Findings:** [Write what the numbers show. Expect throughput to scale roughly
with replicas up to the 4-core limit, then flatten. Note p99 versus median and
whether the 4-replica run hit memory pressure.]

Reproduce with `./locust/run_load_tests.sh`.

## Repository structure

```
├── README.md
├── notebook/coffee_bean_classification.ipynb
├── src/{config,preprocessing,model,prediction,database}.py
├── api/main.py
├── ui/app.py
├── data/{train,test}/          sampled; full set via scripts/download_data.py
├── models/coffee_model.keras
├── locust/
├── scripts/
└── docker/
```

## Known limitations

- **Retraining runs in the API process**, so prediction latency degrades while
  a job is in flight. The fix is a separate worker; it was out of scope here.
  Load tests were run with no retrain active.
- **Space storage is ephemeral.** Uploads and retrained models are lost on
  restart unless `HF_TOKEN` and `HF_MODEL_REPO` are configured.
- **Predictions are only trustworthy on single-bean photos on a plain
  background**, matching the training data. Multi-bean or cluttered images are
  out of distribution.
- **The load test runs locally**, not on the Space — a free Space is a single
  container with no replica control.
````

- [ ] **Step 3: Run the full verification sweep**

```bash
.venv/bin/pytest tests/ -v
docker-compose up -d --build && sleep 45
curl -s localhost:8080/api/health
curl -s localhost:8080/api/status | python3 -m json.tool | head -20
curl -s -o /dev/null -w "%{http_code}\n" localhost:8080/docs
```

Expected: all tests pass, health ok, `"model_loaded": true`, `/docs` returns 200.

- [ ] **Step 4: Confirm the repo is a sane size and nothing large slipped in**

```bash
git count-objects -vH | grep size-pack
git ls-files | xargs du -ch 2>/dev/null | tail -1
git ls-files | awk '{print}' | xargs -I{} du -k {} 2>/dev/null | sort -rn | head -5
```

Expected: pack under ~50MB. If `data/full/` or `best.weights.h5` appear in
`git ls-files`, they were committed before `.gitignore` took effect — remove
them with `git rm --cached` before submitting.

- [ ] **Step 5: Verify the demo images the video will use**

```bash
for f in $(ls data/test/longberry/*.jpg | head -3) $(ls data/test/peaberry/*.jpg | head -3); do
  echo -n "$f -> "
  curl -s -F "file=@$f" localhost:8080/api/predict | \
    python3 -c "import sys,json; d=json.load(sys.stdin); print(d['class'], round(d['confidence'],3))"
done
```

Expected: correct classes with high confidence. Pick the highest-confidence
correct images for the video. Do not use `defect` images — 0.767 recall means
roughly a one-in-four chance of a misclassification on camera.

- [ ] **Step 6: Commit**

```bash
git add README.md notebook/coffee_bean_classification.ipynb
git commit -m "docs: add README with results and fix notebook retrain labels"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §2 Repo structure, housekeeping | 1 |
| §3 Data strategy, sample vs full, eval slice | 1, 5, 12 |
| §4 `config.py` | 1 |
| §4 `preprocessing.py` | 3, 4, 5 |
| §4 `model.py` | 7, 8 |
| §4 `prediction.py` | 6 |
| §4 `database.py` | 2 |
| §5 API surface, triggering, error codes | 9, 10, 11 |
| §6 Retrain pipeline, gate, hot-swap, HF push | 8, 10 |
| §7 UI pages and visualizations | 11, 13 |
| §8 Docker, compose, nginx, Space | 14, 16 |
| §9 Load test methodology | 15 |
| §10 README | 17 |
| §11 Video requirements | 17 Step 5 (demo image verification) |
| §12 Testing strategy | 1-11 (TDD throughout) |

**Type consistency:** `set_model(model, version)`, `predict_image(bytes) -> dict`,
`stage_upload(bytes, batch_id) -> dict`, `evaluate(model, ds) -> dict`,
`retrain(progress_cb, epochs, replay_n) -> dict`, and `promote(model, metrics) -> str`
are used with identical signatures everywhere they appear.

**Known gaps, deliberate:** the video itself (Task 17 Step 5 only verifies the demo
images) and creating the HF Dataset repo are manual steps the user performs.
