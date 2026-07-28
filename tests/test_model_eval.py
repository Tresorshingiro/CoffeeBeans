import numpy as np
import pytest
import tensorflow as tf

from src import config, model as model_module


class PerfectStub:
    """Predicts each sample's true label with full confidence."""

    def __init__(self, labels):
        self.labels = labels

    def predict(self, ds, verbose=0):
        out = np.zeros((len(self.labels), len(config.CLASS_NAMES)),
                       dtype="float32")
        for row, label in enumerate(self.labels):
            out[row, label] = 1.0
        return out


def make_ds(labels):
    images = tf.zeros((len(labels), *config.IMG_SIZE, 3))
    return tf.data.Dataset.from_tensor_slices(
        (images, np.array(labels, dtype="int32"))).batch(2)


def test_evaluate_perfect_predictions():
    labels = [0, 1, 2, 3, 0, 1, 2, 3]
    result = model_module.evaluate(PerfectStub(labels), make_ds(labels))
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["n_samples"] == 8
    assert result["per_class"]["defect"]["recall"] == pytest.approx(1.0)


def test_evaluate_reports_all_four_metrics():
    labels = [0, 1, 2, 3]
    result = model_module.evaluate(PerfectStub(labels), make_ds(labels))
    assert set(result) >= {"accuracy", "loss", "per_class", "confusion_matrix"}
    for class_name in config.CLASS_NAMES:
        assert set(result["per_class"][class_name]) == {
            "precision", "recall", "f1", "support"}


def test_evaluate_confusion_matrix_shape():
    labels = [0, 1, 2, 3]
    result = model_module.evaluate(PerfectStub(labels), make_ds(labels))
    matrix = result["confusion_matrix"]
    assert len(matrix) == 4
    assert all(len(row) == 4 for row in matrix)


def test_evaluate_imperfect_predictions():
    true_labels = [0, 0, 1, 1]
    predicted = [0, 1, 1, 1]
    result = model_module.evaluate(PerfectStub(predicted), make_ds(true_labels))
    assert result["accuracy"] == pytest.approx(0.75)


def test_freeze_batchnorm():
    built = model_module.build_model()
    model_module.freeze_batchnorm(built)
    bn_layers = [layer for layer in built.get_layer("mobilenetv2_1.00_224").layers
                 if isinstance(layer, tf.keras.layers.BatchNormalization)]
    assert bn_layers
    assert all(layer.trainable is False for layer in bn_layers)
