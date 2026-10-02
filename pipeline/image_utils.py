"""Image utility functions for validation, I/O, and metadata."""
import os
import uuid
import mimetypes
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import logging

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
ALLOWED_MIMES = {'image/jpeg', 'image/png', 'image/webp'}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


def allowed_file(filename):
    """Check if a filename has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image_file(file) -> tuple:
    """Validate an uploaded file for extension, MIME type, and size."""
    if not file or not file.filename:
        return False, "No file provided"

    ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Extension not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"

    mimetype = getattr(file, 'mimetype', None)
    if not mimetype:
        mimetype, _ = mimetypes.guess_type(file.filename)

    if mimetype not in ALLOWED_MIMES:
        return False, f"MIME type '{mimetype}' not allowed"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size > MAX_FILE_SIZE:
        return False, f"File too large ({file_size} bytes). Max: {MAX_FILE_SIZE} bytes"

    if file_size == 0:
        return False, "File is empty"

    return True, None


def read_image_from_upload(file) -> np.ndarray:
    """Read image from a Werkzeug FileStorage to BGR numpy array."""
    file_bytes = np.frombuffer(file.read(), np.uint8)
    file.seek(0)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    return img_bgr


def read_image_from_bytes(data: bytes) -> np.ndarray:
    """Read image from raw bytes to BGR numpy array."""
    file_bytes = np.frombuffer(data, np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)


def save_image(img_bgr, output_path, format='png') -> bool:
    """Save a BGR numpy array to disk."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        params = []
        if format in ('jpg', 'jpeg'):
            params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        elif format == 'png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        elif format == 'webp':
            params = [cv2.IMWRITE_WEBP_QUALITY, 95]
        return cv2.imwrite(output_path, img_bgr, params)
    except Exception as e:
        logger.error(f"Failed to save image to {output_path}: {e}")
        return False


def get_image_info(img: np.ndarray) -> dict:
    """Return shape, dtype, and channel count of an image."""
    if img is None:
        return {}
    shape = img.shape
    channels = shape[2] if len(shape) == 3 else 1
    return {
        'shape': shape,
        'dtype': str(img.dtype),
        'channels': channels,
        'height': shape[0],
        'width': shape[1],
    }


def sanitize_filename(filename: str) -> str:
    """Create a safe filename with a UUID prefix."""
    safe_name = secure_filename(filename)
    return f"{uuid.uuid4().hex[:8]}_{safe_name}"


def generate_output_id() -> str:
    """Generate a unique UUID string for output identification."""
    return str(uuid.uuid4())
