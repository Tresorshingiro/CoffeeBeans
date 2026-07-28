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
