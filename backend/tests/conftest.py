"""Shared fixtures: every test session gets its own data dir, so nothing touches the repo."""
import io
import os
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image

_DATA_DIR = tempfile.mkdtemp(prefix="visionqc-test-")
os.environ["VISIONQC_DATA_DIR"] = _DATA_DIR
os.environ.pop("VISIONQC_DB_PATH", None)
os.environ.pop("VISIONQC_API_KEY", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_png(color=(200, 180, 120), size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def png_bytes():
    return make_png()


@pytest.fixture
def data_dir():
    return Path(_DATA_DIR)
