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
