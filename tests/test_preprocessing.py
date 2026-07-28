import pytest

from src import config, preprocessing


def test_decode_jpeg(jpeg_bytes):
    img = preprocessing.decode_image_bytes(jpeg_bytes())
    assert tuple(img.shape) == (*config.IMG_SIZE, 3)


def test_decode_png(png_bytes):
    # The notebook used decode_jpeg, which raises on PNG. User uploads
    # will contain PNGs, so decode_image is required here.
    img = preprocessing.decode_image_bytes(png_bytes())
    assert tuple(img.shape) == (*config.IMG_SIZE, 3)


def test_decode_rejects_garbage():
    with pytest.raises(Exception):
        preprocessing.decode_image_bytes(b"this is not an image")


def test_list_images_assigns_labels_from_class_names(image_tree):
    root = image_tree("train", {"defect": 2, "longberry": 3})
    paths, labels = preprocessing.list_images(root)
    assert len(paths) == 5
    assert sorted(set(labels)) == [0, 1]


def test_list_images_partial_classes_keep_global_indices(image_tree):
    # Regression test for the label-ordering bug. Only defect (0) and
    # premium (3) are present; premium must stay 3, not collapse to 1.
    root = image_tree("partial", {"defect": 2, "premium": 2})
    paths, labels = preprocessing.list_images(root)
    assert sorted(set(labels)) == [0, 3]


def test_list_images_ignores_unknown_directories(image_tree):
    root = image_tree("mixed", {"defect": 2})
    (root / "not_a_class").mkdir()
    (root / "not_a_class" / "x.jpg").write_bytes(b"junk")
    paths, labels = preprocessing.list_images(root)
    assert len(paths) == 2


def test_dataset_from_paths_shapes(image_tree):
    root = image_tree("train", {"defect": 4, "premium": 4})
    paths, labels = preprocessing.list_images(root)
    ds = preprocessing.dataset_from_paths(paths, labels, batch=4)
    images, batch_labels = next(iter(ds))
    assert tuple(images.shape) == (4, *config.IMG_SIZE, 3)
    assert tuple(batch_labels.shape) == (4,)


def test_load_dataset_preserves_order_when_not_shuffled(image_tree):
    root = image_tree("test", {"defect": 4, "longberry": 4})
    ds = preprocessing.load_dataset(root, shuffle=False)
    labels = [int(v) for _, batch in ds for v in batch]
    assert labels == sorted(labels)


def test_dataset_from_paths_empty():
    # Empty input must yield an empty dataset, not crash with TypeError.
    # This prevents confusing TensorFlow errors in downstream code.
    ds = preprocessing.dataset_from_paths([], [])
    assert list(ds) == []
