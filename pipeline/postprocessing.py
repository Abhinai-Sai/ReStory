import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def postprocess_image(img_bgr: np.ndarray, original_format='png') -> np.ndarray:
    """
    Post-processes the image by ensuring valid uint8 range, 
    3-channel BGR, and optional denoising.
    """
    if img_bgr is None:
        raise ValueError("Empty image provided to postprocessing")

    # Ensure valid uint8 range
    img_bgr = np.clip(img_bgr, 0, 255)
    
    if img_bgr.dtype != np.uint8:
        img_bgr = img_bgr.astype(np.uint8)

    # Ensure 3-channel BGR
    if len(img_bgr.shape) == 2 or (len(img_bgr.shape) == 3 and img_bgr.shape[2] == 1):
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif len(img_bgr.shape) == 3 and img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    # Apply subtle adaptive detail sharpening (unsharp mask) for enhanced clarity
    gaussian = cv2.GaussianBlur(img_bgr, (0, 0), 2.0)
    sharpened = cv2.addWeighted(img_bgr, 1.25, gaussian, -0.25, 0)
    img_bgr = np.clip(sharpened, 0, 255).astype(np.uint8)

    return img_bgr

def encode_image_to_bytes(img_bgr: np.ndarray, format='png') -> bytes:
    """Encodes a BGR image to bytes in the specified format."""
    success, encoded_img = cv2.imencode(f'.{format.lower()}', img_bgr)
    if not success:
        raise RuntimeError(f"Failed to encode image to {format}")
    return encoded_img.tobytes()

def get_output_format(original_filename: str) -> str:
    """Determines the appropriate output format based on the original filename."""
    if not original_filename:
        return 'png'
        
    ext = original_filename.rsplit('.', 1)[-1].lower() if '.' in original_filename else ''
    
    if ext in ['jpg', 'jpeg']:
        return 'jpg'
    elif ext in ['webp']:
        return 'webp'
    return 'png'
