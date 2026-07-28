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
