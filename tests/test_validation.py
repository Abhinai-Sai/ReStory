"""Tests for image validation utilities."""
import io
import os
import pytest
import numpy as np
import cv2

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.image_utils import validate_image_file, allowed_file, save_image, generate_output_id
from pipeline.preprocessing import preprocess_image
from pipeline.postprocessing import postprocess_image, get_output_format


class FakeFile:
    """Mimics a Werkzeug FileStorage for testing."""
    def __init__(self, filename, data, mimetype=None):
        self.filename = filename
        self.mimetype = mimetype
        self._data = data
        self._pos = 0

    def read(self):
        return self._data

    def seek(self, offset, whence=0):
        if whence == 0:
            self._pos = offset
        elif whence == 2:
            self._pos = len(self._data) + offset

    def tell(self):
        return len(self._data) if self._pos == 0 else self._pos


def make_test_image(width=100, height=100, channels=3):
    """Create a simple test image as bytes."""
    img = np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
    _, buf = cv2.imencode('.png', img)
    return buf.tobytes()


# --- Validation tests ---

class TestValidation:
    def test_valid_image(self):
        data = make_test_image()
        f = FakeFile('test.png', data, mimetype='image/png')
        valid, err = validate_image_file(f)
        assert valid is True
        assert err is None

    def test_invalid_extension(self):
        data = make_test_image()
        f = FakeFile('test.bmp', data, mimetype='image/bmp')
        valid, err = validate_image_file(f)
        assert valid is False
        assert 'Extension' in err or 'not allowed' in err.lower()

    def test_invalid_mime_type(self):
        data = make_test_image()
        f = FakeFile('test.png', data, mimetype='application/pdf')
        valid, err = validate_image_file(f)
        assert valid is False
        assert 'MIME' in err or 'not allowed' in err.lower()

    def test_oversized_file(self):
        # Create data bigger than 20MB
        data = b'x' * (21 * 1024 * 1024)
        f = FakeFile('big.png', data, mimetype='image/png')
        valid, err = validate_image_file(f)
        assert valid is False
        assert 'large' in err.lower() or 'size' in err.lower()

    def test_missing_file(self):
        valid, err = validate_image_file(None)
        assert valid is False

    def test_empty_filename(self):
        f = FakeFile('', b'data', mimetype='image/png')
        valid, err = validate_image_file(f)
        assert valid is False

    def test_empty_file(self):
        f = FakeFile('test.png', b'', mimetype='image/png')
        valid, err = validate_image_file(f)
        assert valid is False

    def test_allowed_file_function(self):
        assert allowed_file('test.jpg') is True
        assert allowed_file('test.jpeg') is True
        assert allowed_file('test.png') is True
        assert allowed_file('test.webp') is True
        assert allowed_file('test.bmp') is False
        assert allowed_file('test.gif') is False
        assert allowed_file('noext') is False


# --- Preprocessing tests ---

class TestPreprocessing:
    def test_rgb_image(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = preprocess_image(img)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_grayscale_image(self):
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = preprocess_image(img)
        assert len(result.shape) == 3
        assert result.shape[2] == 3

    def test_rgba_image(self):
        img = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        result = preprocess_image(img)
        assert result.shape[2] == 3

    def test_tiny_image_raises(self):
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="too small"):
            preprocess_image(img)

    def test_large_image_resized(self):
        img = np.random.randint(0, 255, (5000, 3000, 3), dtype=np.uint8)
        result = preprocess_image(img)
        assert max(result.shape[:2]) <= 2048

    def test_float_image(self):
        img = np.random.rand(100, 100, 3).astype(np.float32)
        result = preprocess_image(img)
        assert result.dtype == np.uint8


# --- Postprocessing tests ---

class TestPostprocessing:
    def test_valid_image(self):
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        result = postprocess_image(img)
        assert result.dtype == np.uint8
        assert result.shape[2] == 3

    def test_clip_values(self):
        img = np.array([[[300, -10, 128]]], dtype=np.float64)
        result = postprocess_image(img)
        assert result.max() <= 255
        assert result.min() >= 0

    def test_none_raises(self):
        with pytest.raises(ValueError):
            postprocess_image(None)

    def test_get_output_format(self):
        assert get_output_format('photo.jpg') == 'jpg'
        assert get_output_format('photo.jpeg') == 'jpg'
        assert get_output_format('photo.png') == 'png'
        assert get_output_format('photo.webp') == 'webp'
        assert get_output_format('') == 'png'


# --- Utility tests ---

class TestUtils:
    def test_generate_output_id(self):
        id1 = generate_output_id()
        id2 = generate_output_id()
        assert isinstance(id1, str)
        assert len(id1) > 0
        assert id1 != id2

    def test_save_image(self, tmp_path):
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        path = str(tmp_path / 'test_output.png')
        result = save_image(img, path, format='png')
        assert result is True
        assert os.path.exists(path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
