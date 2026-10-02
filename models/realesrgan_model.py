"""Real-ESRGAN model wrapper for general image restoration/super-resolution."""
import os
import logging
import torch
import urllib.request
import numpy as np
import cv2

logger = logging.getLogger(__name__)

REALESRGAN_URLS = [
    'https://huggingface.co/ckpt/RealESRGAN_x4plus/resolve/main/RealESRGAN_x4plus.pth',
    'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
]
WEIGHTS_FILENAME = 'RealESRGAN_x4plus.pth'


class RealESRGANWrapper:
    """Wrapper around the Real-ESRGAN inference pipeline."""

    def __init__(self, device=None, weights_dir='weights'):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights_dir = os.path.abspath(weights_dir)
        self.weights_path = os.path.join(self.weights_dir, WEIGHTS_FILENAME)
        self.upsampler = None
        self._load_model()

    def download_weights(self):
        """Download model weights if they don't exist."""
        os.makedirs(self.weights_dir, exist_ok=True)

        if not os.path.exists(self.weights_path) or os.path.getsize(self.weights_path) < 1_000_000:
            logger.info(f"Downloading RealESRGAN_x4plus weights to {self.weights_path}...")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            success = False
            last_err = None

            for url in REALESRGAN_URLS:
                try:
                    logger.info(f"Attempting download from {url}...")
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=120) as response, open(self.weights_path, 'wb') as out_file:
                        data = response.read()
                        out_file.write(data)

                    file_size = os.path.getsize(self.weights_path)
                    logger.info(f"Download finished. Size: {file_size / 1024 / 1024:.1f} MB")
                    if file_size >= 1_000_000:
                        success = True
                        break
                    else:
                        logger.warning("Downloaded file too small, removing and trying next mirror...")
                        if os.path.exists(self.weights_path):
                            os.remove(self.weights_path)
                except Exception as e:
                    logger.warning(f"Failed to download from {url}: {e}")
                    last_err = e

            if not success:
                if os.path.exists(self.weights_path):
                    os.remove(self.weights_path)
                raise RuntimeError(f"Failed to download Real-ESRGAN weights from all mirrors: {last_err}")
        else:
            logger.info(f"Weights already exist at {self.weights_path}")

    def _load_model(self):
        """Initialize the Real-ESRGAN model."""
        self.download_weights()

        try:
            from realesrgan import RealESRGANer
        except ImportError:
            logger.error("realesrgan package not installed. Install with: pip install realesrgan")
            raise

        # Import RRDBNet from local architecture file
        from .rrdbnet_arch import RRDBNet

        logger.info("Initializing RRDBNet architecture...")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

        half_precision = self.device.type == 'cuda'
        # On CPU, disable tiling (tile=0) to run in 1 single pass for maximum speed.
        # On GPU, use 512 tile size to prevent VRAM OOM.
        tile_size = 512 if self.device.type == 'cuda' else 0

        logger.info(f"Initializing RealESRGANer (half={half_precision}, tile={tile_size}, device={self.device})...")
        try:
            self.upsampler = RealESRGANer(
                scale=4,
                model_path=self.weights_path,
                model=model,
                tile=tile_size,
                tile_pad=10,
                pre_pad=0,
                half=half_precision,
                device=self.device
            )
            logger.info("Real-ESRGAN model loaded successfully.")
        except Exception as e:
            logger.error(f"Error initializing RealESRGANer: {e}")
            raise

    @torch.inference_mode()
    def enhance(self, img_bgr, outscale=4):
        """Enhance an image using Real-ESRGAN.

        Args:
            img_bgr: Input BGR numpy array (uint8).
            outscale: Output upscaling factor.

        Returns:
            Enhanced BGR numpy array (uint8).
        """
        if self.upsampler is None:
            raise RuntimeError("Real-ESRGAN model is not loaded.")

        h, w = img_bgr.shape[:2]
        original_tile = getattr(self.upsampler, 'tile_size', 0)

        # On CPU, process untiled for speed unless image exceeds 1024px
        if self.device.type == 'cpu' and (h > 1024 or w > 1024):
            self.upsampler.tile_size = 512

        logger.info(f"Enhancing image ({w}x{h}) with Real-ESRGAN (tile={getattr(self.upsampler, 'tile_size', 0)}, outscale={outscale})...")
        try:
            output, _ = self.upsampler.enhance(img_bgr, outscale=outscale)
            logger.info(f"Enhancement complete — output shape: {output.shape}")
            return output
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and hasattr(self.upsampler, 'tile_size'):
                logger.warning("GPU OOM — retrying with smaller tile size...")
                self.upsampler.tile_size = max(64, original_tile // 2 if original_tile > 0 else 256)
                try:
                    output, _ = self.upsampler.enhance(img_bgr, outscale=outscale)
                    return output
                except Exception as retry_e:
                    logger.error(f"Retry also failed: {retry_e}")
                    raise
            raise
        finally:
            if hasattr(self.upsampler, 'tile_size'):
                self.upsampler.tile_size = original_tile
