"""Lightweight face detection for pre-screening images."""
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def _get_face_detector():
    """Create a face detector compatible with OpenCV 4.x and 5.x."""
    # OpenCV 5.x uses FaceDetectorYN (DNN-based)
    if hasattr(cv2, 'FaceDetectorYN_create') or hasattr(cv2, 'FaceDetectorYN'):
        try:
            # FaceDetectorYN needs a model file; use a lightweight approach instead
            # Fall back to simple skin-tone heuristic for pre-screening
            return None
        except Exception:
            pass

    # OpenCV 4.x uses CascadeClassifier
    if hasattr(cv2, 'CascadeClassifier'):
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                return cascade
        except Exception:
            pass

    return None


_detector = _get_face_detector()


def detect_faces(img_bgr: np.ndarray) -> list:
    """Detect faces in a BGR image.

    Returns a list of face bounding boxes. If the detector is unavailable,
    returns an empty list (GFPGAN will still attempt its own detection).
    """
    if img_bgr is None:
        return []

    # If we have a CascadeClassifier (OpenCV 4.x)
    if _detector is not None and hasattr(_detector, 'detectMultiScale'):
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = _detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            face_list = list(faces) if len(faces) > 0 else []
            logger.info(f"CascadeClassifier detected {len(face_list)} face(s)")
            return face_list
        except Exception as e:
            logger.warning(f"CascadeClassifier detection failed: {e}")

    # Fallback: simple skin-tone heuristic (rough pre-screen)
    # This doesn't need a model file and works on any OpenCV version.
    # GFPGAN does its own precise detection, so false positives are fine.
    try:
        return _skin_tone_heuristic(img_bgr)
    except Exception as e:
        logger.warning(f"Skin-tone heuristic failed: {e}")
        return []


def _skin_tone_heuristic(img_bgr: np.ndarray) -> list:
    """Very rough check for skin-tone regions that might contain faces.

    This is intentionally permissive — it's better to let GFPGAN try and
    find no faces than to miss a face-containing image.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Broad skin-tone range in HSV
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    skin_ratio = np.count_nonzero(mask) / mask.size

    if skin_ratio > 0.05:  # >5% skin-tone pixels
        logger.info(f"Skin-tone heuristic: {skin_ratio:.1%} skin pixels — likely has faces")
        return [True]  # Signal that faces are likely present
    return []


def has_faces(img_bgr: np.ndarray) -> bool:
    """Check whether an image likely contains faces."""
    faces = detect_faces(img_bgr)
    return len(faces) > 0

