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
