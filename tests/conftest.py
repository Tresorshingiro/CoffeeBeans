import io

import pytest
from PIL import Image


def _png_bytes(color=(120, 140, 90), size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(color=(120, 140, 90), size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def png_bytes():
    return _png_bytes


@pytest.fixture
def jpeg_bytes():
    return _jpeg_bytes


@pytest.fixture
def image_tree(tmp_path):
    """Builds a directory tree of JPEGs. Returns (root, {class: count})."""
    def _build(root_name, counts):
        root = tmp_path / root_name
        for cls, n in counts.items():
            d = root / cls
            d.mkdir(parents=True)
            for i in range(n):
                (d / f"{i}.jpg").write_bytes(_jpeg_bytes())
        return root
    return _build
