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
MAX_DECOMPRESSED_BYTES = 500 * 1024 * 1024
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
