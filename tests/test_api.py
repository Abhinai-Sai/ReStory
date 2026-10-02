"""Tests for Flask API endpoints."""
import os
import sys
import io
import pytest
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from config import Config


class TestConfig(Config):
    TESTING = True
    UPLOAD_FOLDER = os.path.join(Config.BASE_DIR, 'test_uploads')
    OUTPUT_FOLDER = os.path.join(Config.BASE_DIR, 'test_outputs')


def make_test_image_bytes(width=100, height=100, fmt='.png'):
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    _, buf = cv2.imencode(fmt, img)
    return buf.tobytes()


@pytest.fixture
def client():
    app = create_app(TestConfig)
    with app.test_client() as c:
        yield c
    # Cleanup test dirs
    import shutil
    for d in [TestConfig.UPLOAD_FOLDER, TestConfig.OUTPUT_FOLDER]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get('/api/health')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'healthy'
        assert 'device' in data
        assert 'timestamp' in data

    def test_health_has_device(self, client):
        resp = client.get('/api/health')
        data = resp.get_json()
        assert data['device'] in ('cpu', 'cuda')


class TestRestoreEndpoint:
    def test_no_file(self, client):
        resp = client.post('/api/restore')
        assert resp.status_code == 400
        data = resp.get_json()
        assert data['success'] is False

    def test_empty_filename(self, client):
        resp = client.post('/api/restore', data={
            'image': (io.BytesIO(b''), '')
        }, content_type='multipart/form-data')
        assert resp.status_code == 400

    def test_invalid_extension(self, client):
        resp = client.post('/api/restore', data={
            'image': (io.BytesIO(b'fake'), 'test.bmp')
        }, content_type='multipart/form-data')
        assert resp.status_code == 400
        data = resp.get_json()
        assert data['success'] is False

    def test_response_format(self, client):
        """Ensure error responses have proper JSON structure."""
        resp = client.post('/api/restore')
        data = resp.get_json()
        assert 'success' in data
        assert 'error' in data or 'message' in data


class TestResultEndpoint:
    def test_not_found(self, client):
        resp = client.get('/api/result/nonexistent-id')
        assert resp.status_code == 404

    def test_path_traversal_blocked(self, client):
        resp = client.get('/api/result/../../../etc/passwd')
        assert resp.status_code in (400, 404)  # Flask normalizes ../ → 404 is also safe


class TestDownloadEndpoint:
    def test_not_found(self, client):
        resp = client.get('/api/download/nonexistent-id')
        assert resp.status_code == 404

    def test_path_traversal_blocked(self, client):
        resp = client.get('/api/download/../../../etc/passwd')
        assert resp.status_code in (400, 404)  # Flask normalizes ../ → 404 is also safe


class TestHomepage:
    def test_homepage_loads(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        assert b'AI Art Restoration' in resp.data


class TestErrorHandlers:
    def test_404_json(self, client):
        resp = client.get('/api/nonexistent')
        assert resp.status_code == 404
        data = resp.get_json()
        assert data['success'] is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
