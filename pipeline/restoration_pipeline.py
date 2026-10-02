"""Main restoration pipeline orchestrator."""
import os
import time
import logging
from typing import Dict, Any

import cv2
import numpy as np

from .image_utils import read_image_from_bytes, save_image, generate_output_id, get_image_info
from .preprocessing import preprocess_image
from .face_processing import has_faces
from .postprocessing import postprocess_image, get_output_format

logger = logging.getLogger(__name__)


class RestorationPipeline:
    """Orchestrates the complete art restoration workflow."""

    def __init__(self, model_manager):
        self.model_manager = model_manager

    def restore(self, input_path: str, output_dir: str = "outputs", outscale: int = 4) -> Dict[str, Any]:
        """
        Full restoration pipeline:
        validate → decode → preprocess → Real-ESRGAN → detect faces
        → GFPGAN (if applicable) → postprocess → save

        Args:
            input_path: Path to the uploaded image file on disk.
            output_dir: Directory to save the restored output.
            outscale: Output upscaling factor (2 for Fast Mode, 4 for High Quality).

        Returns:
            Dict with output_id, output_path, success status, and details.
        """
        start_time = time.time()
        timing = {}

        try:
            # 1. Read / decode image
            t0 = time.time()
            img_bgr = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if img_bgr is None:
                return {"success": False, "error": "Failed to decode image — file may be corrupted"}
            timing['decode'] = round(time.time() - t0, 3)
            logger.info(f"Decoded image: {get_image_info(img_bgr)}")

            # Remember the original format for output
            ext = os.path.splitext(input_path)[1].lower().lstrip('.')
            out_format = ext if ext in ('jpg', 'jpeg', 'png', 'webp') else 'png'

            # 2. Preprocess
            t0 = time.time()
            img_bgr = preprocess_image(img_bgr)
            timing['preprocess'] = round(time.time() - t0, 3)

            # 3. General restoration / super-resolution (Real-ESRGAN)
            t0 = time.time()
            try:
                realesrgan = self.model_manager.get_realesrgan()
                enhanced_img = realesrgan.enhance(img_bgr, outscale=outscale)
                logger.info(f"Real-ESRGAN enhancement complete (outscale={outscale}): {get_image_info(enhanced_img)}")
            except Exception as e:
                logger.warning(f"Real-ESRGAN unavailable ({e}), using high-quality Lanczos4 super-resolution fallback")
                h, w = img_bgr.shape[:2]
                enhanced_img = cv2.resize(img_bgr, (w * outscale, h * outscale), interpolation=cv2.INTER_LANCZOS4)
                gaussian = cv2.GaussianBlur(enhanced_img, (0, 0), 2.0)
                enhanced_img = cv2.addWeighted(enhanced_img, 1.3, gaussian, -0.3, 0)
            timing['realesrgan'] = round(time.time() - t0, 3)

            # 4. Detect faces
            t0 = time.time()
            faces_present = has_faces(enhanced_img)
            timing['face_detect'] = round(time.time() - t0, 3)
            logger.info(f"Face detection: {'faces found' if faces_present else 'no faces'}")

            # 5. GFPGAN face restoration (if applicable)
            if faces_present:
                t0 = time.time()
                try:
                    gfpgan = self.model_manager.get_gfpgan()
                    realesrgan_model = self.model_manager.get_realesrgan()
                    face_result, had_faces = gfpgan.enhance_faces(
                        enhanced_img, realesrgan_model=realesrgan_model
                    )
                    if had_faces and face_result is not None:
                        enhanced_img = face_result
                        logger.info("GFPGAN face restoration applied successfully.")
                    else:
                        logger.info("GFPGAN returned no face results, using Real-ESRGAN output.")
                except Exception as e:
                    logger.warning(f"GFPGAN failed, falling back to Real-ESRGAN result: {e}")
                timing['gfpgan'] = round(time.time() - t0, 3)

            # 6. Post-process
            t0 = time.time()
            final_img = postprocess_image(enhanced_img, original_format=out_format)
            timing['postprocess'] = round(time.time() - t0, 3)

            # 7. Save output
            t0 = time.time()
            output_id = generate_output_id()
            out_ext = 'jpg' if out_format == 'jpeg' else out_format
            output_path = os.path.join(output_dir, f"{output_id}.{out_ext}")
            os.makedirs(output_dir, exist_ok=True)

            if not save_image(final_img, output_path, format=out_ext):
                return {"success": False, "error": "Failed to save restored image"}
            timing['save'] = round(time.time() - t0, 3)

            total_time = round(time.time() - start_time, 2)
            timing['total'] = total_time

            device_used = str(self.model_manager.get_device())

            result = {
                "success": True,
                "output_id": output_id,
                "output_path": output_path,
                "details": {
                    "has_faces": faces_present,
                    "faces_detected": 1 if faces_present else 0,
                    "device": device_used,
                    "timing": timing,
                    "output_format": out_ext,
                    "output_resolution": list(final_img.shape[:2]),
                },
            }
            logger.info(f"Restoration complete in {total_time}s → {output_path}")
            return result

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
