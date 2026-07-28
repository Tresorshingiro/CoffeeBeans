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
