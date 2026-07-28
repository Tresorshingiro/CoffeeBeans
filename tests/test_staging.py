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


def test_rejects_member_exceeding_decompressed_limit(isolated, jpeg_bytes, monkeypatch):
    """Archive exceeding decompressed limit is rejected."""
    # Set a tiny limit so a normal JPEG exceeds it
    # Use a module-level patch since preprocessing imports config at module load time
    import src.preprocessing as preproc_module
    monkeypatch.setattr(preproc_module.config, "MAX_DECOMPRESSED_BYTES", 100)
    blob = make_zip({"defect/a.jpg": jpeg_bytes()})
    with pytest.raises(ValueError, match="exceeds maximum decompressed size"):
        preprocessing.stage_upload(blob, "batch-bomb")


def test_normal_small_archive_still_succeeds(jpeg_bytes):
    """Verify the decompression guard doesn't regress normal uploads."""
    blob = make_zip({"defect/a.jpg": jpeg_bytes((1, 1, 1))})
    result = preprocessing.stage_upload(blob, "batch-normal")
    assert result["total_accepted"] == 1
    assert result["accepted"]["defect"] == 1
