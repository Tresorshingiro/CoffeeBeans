from src import config, preprocessing


def test_sample_replay_is_stratified(image_tree, monkeypatch):
    root = image_tree("train", {c: 50 for c in config.CLASS_NAMES})
    monkeypatch.setattr(config, "train_dir", lambda: root)
    paths, labels = preprocessing.sample_replay(40)
    assert len(paths) == 40
    for index in range(len(config.CLASS_NAMES)):
        assert labels.count(index) == 10


def test_sample_replay_handles_small_classes(image_tree, monkeypatch):
    root = image_tree("train", {"defect": 3, "longberry": 50,
                                "peaberry": 50, "premium": 50})
    monkeypatch.setattr(config, "train_dir", lambda: root)
    paths, labels = preprocessing.sample_replay(40)
    assert labels.count(0) == 3      # capped by what exists
    assert labels.count(1) == 10


def test_sample_replay_is_deterministic(image_tree, monkeypatch):
    root = image_tree("train", {c: 50 for c in config.CLASS_NAMES})
    monkeypatch.setattr(config, "train_dir", lambda: root)
    first, _ = preprocessing.sample_replay(20)
    second, _ = preprocessing.sample_replay(20)
    assert first == second


def test_eval_slice_is_fixed_and_stratified(image_tree, monkeypatch):
    root = image_tree("test", {c: 200 for c in config.CLASS_NAMES})
    monkeypatch.setattr(config, "test_dir", lambda: root)
    monkeypatch.setattr(config, "EVAL_SLICE_SIZE", 40)
    first, labels = preprocessing.eval_slice()
    second, _ = preprocessing.eval_slice()
    assert first == second               # same slice every run
    assert len(first) == 40
    for index in range(len(config.CLASS_NAMES)):
        assert labels.count(index) == 10
