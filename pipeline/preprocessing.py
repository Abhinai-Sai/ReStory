import cv2
import numpy as np
import logging
from .image_utils import get_image_info

logger = logging.getLogger(__name__)

def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocesses the image for the restoration pipeline.
    Handles grayscale conversion to 3-channel, RGBA to RGB conversion,
    and validates/resizes image dimensions.
    """
    logger.info(f"Image info before preprocessing: {get_image_info(img_bgr)}")

    if img_bgr is None:
        raise ValueError("Empty image provided")

    # Handle RGBA to RGB
    if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    
    # Handle Grayscale to BGR
    elif len(img_bgr.shape) == 2 or (len(img_bgr.shape) == 3 and img_bgr.shape[2] == 1):
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    # Validate dimensions (min 16x16, max 4096x4096)
    h, w = img_bgr.shape[:2]
    if h < 16 or w < 16:
        raise ValueError(f"Image dimensions ({w}x{h}) are too small. Minimum 16x16 required.")

    # Cap max input dimension to 480px to guarantee 1-pass execution under 15s on CPU while yielding 1920px HD output
    MAX_DIM = 480
    if h > MAX_DIM or w > MAX_DIM:
        scale = float(MAX_DIM) / max(h, w)
        new_w, new_h = max(16, int(w * scale)), max(16, int(h * scale))
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(f"Pre-scaled image from {w}x{h} to {new_w}x{new_h} for high speed 1-pass processing")

    # Ensure uint8 dtype
    if img_bgr.dtype != np.uint8:
        if img_bgr.dtype in [np.float32, np.float64]:
            img_bgr = (img_bgr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img_bgr = img_bgr.astype(np.uint8)

    logger.info(f"Image info after preprocessing: {get_image_info(img_bgr)}")
    return img_bgr
