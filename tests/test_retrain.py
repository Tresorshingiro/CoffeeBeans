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


class FineTuneStub:
    """Stands in for a loaded Keras model so _run_finetune needs no real graph."""

    layers: list = []

    def compile(self, **kwargs):
        self.compiled = True

    def fit(self, ds, epochs=None, verbose=0, callbacks=None):
        self.fit_epochs = epochs


def _record_loaded_paths(monkeypatch):
    """Patch load_model to record the path it is asked for. Returns the list."""
    import tensorflow as tf

    loaded: list[str] = []

    def fake_load_model(path, *args, **kwargs):
        loaded.append(str(path))
        return FineTuneStub()

    monkeypatch.setattr(tf.keras.models, "load_model", fake_load_model)
    return loaded


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


def test_finetune_starts_from_registered_champion(monkeypatch):
    # A promoted model lives at a timestamped path, not CHAMPION_PATH. The
    # second retrain must build on it, or improvements never compound and the
    # gate compares a baseline-derived candidate against a promoted champion.
    promoted = config.MODELS_DIR / "coffee_model_20260728_120000.keras"
    promoted.touch()
    database.register_model("coffee_model_20260728_120000", str(promoted), 0.88)

    loaded = _record_loaded_paths(monkeypatch)
    model_module._run_finetune(["a.jpg"], [0], epochs=1, emit=lambda _: None)

    assert loaded == [str(promoted)]


def test_finetune_falls_back_to_baseline_when_registry_is_empty(monkeypatch):
    # Fresh install before register_baseline.py has run.
    loaded = _record_loaded_paths(monkeypatch)
    model_module._run_finetune(["a.jpg"], [0], epochs=1, emit=lambda _: None)

    assert loaded == [str(config.CHAMPION_PATH)]


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
