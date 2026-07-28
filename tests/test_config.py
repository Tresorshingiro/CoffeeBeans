from src import config


def test_class_names_are_alphabetical():
    # Keras derives labels alphabetically from directory names. If CLASS_NAMES
    # ever diverges from sorted order, every label index silently shifts.
    assert config.CLASS_NAMES == sorted(config.CLASS_NAMES)


def test_class_names_match_train_directories():
    found = sorted(p.name for p in config.train_dir().iterdir() if p.is_dir())
    assert found == config.CLASS_NAMES


def test_train_dir_prefers_full_dataset_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "FULL_DIR", tmp_path / "full")
    (tmp_path / "train").mkdir()
    assert config.train_dir() == tmp_path / "train"

    full_train = tmp_path / "full" / "train"
    full_train.mkdir(parents=True)
    (full_train / "defect").mkdir()
    assert config.train_dir() == full_train
