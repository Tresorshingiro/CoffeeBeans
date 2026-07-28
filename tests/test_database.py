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
