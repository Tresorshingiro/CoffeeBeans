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
