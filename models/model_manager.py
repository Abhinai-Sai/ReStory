"""Model management with lazy loading and device detection."""
import os
import logging
import threading
import gc
import torch

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton-style manager for AI models with lazy loading."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, config=None):
        self.config = config or {}
        self.device = torch.device(
            self.config.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.weights_dir = self.config.get('MODEL_DIR', 'weights')

        # Optimize PyTorch CPU thread utilization
        if self.device.type == 'cpu':
            num_cores = os.cpu_count() or 4
            torch.set_num_threads(num_cores)
            try:
                torch.set_num_interop_threads(2)
            except Exception:
                pass
            logger.info(f"PyTorch CPU threads set to {num_cores}")

        logger.info(f"ModelManager initialized — device: {self.device}, weights: {self.weights_dir}")

        self._realesrgan = None
        self._gfpgan = None
        self._realesrgan_lock = threading.Lock()
        self._gfpgan_lock = threading.Lock()

    def preload_async(self):
        """Asynchronously pre-load models in a background thread to eliminate first-request cold-starts."""
        def _loader():
            logger.info("Background model pre-loading started...")
            try:
                self.get_realesrgan()
                self.get_gfpgan()
                logger.info("Background model pre-loading completed successfully.")
            except Exception as e:
                logger.warning(f"Background model pre-loading failed (will retry on request): {e}")

        thread = threading.Thread(target=_loader, daemon=True, name="ModelPreloader")
        thread.start()

    @property
    def models(self):
        """Return dict of loaded model names for health check."""
        loaded = {}
        if self._realesrgan is not None:
            loaded['realesrgan'] = True
        if self._gfpgan is not None:
            loaded['gfpgan'] = True
        return loaded

    def get_device(self):
        return self.device

    def get_realesrgan(self):
        with self._realesrgan_lock:
            if self._realesrgan is None:
                logger.info("Lazily loading Real-ESRGAN model...")
                try:
                    from .realesrgan_model import RealESRGANWrapper
                    self._realesrgan = RealESRGANWrapper(
                        device=self.device,
                        weights_dir=self.weights_dir
                    )
                    logger.info("Real-ESRGAN model loaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to load Real-ESRGAN model: {e}")
                    raise
            return self._realesrgan

    def get_gfpgan(self):
        with self._gfpgan_lock:
            if self._gfpgan is None:
                logger.info("Lazily loading GFPGAN model...")
                try:
                    from .gfpgan_model import GFPGANWrapper
                    self._gfpgan = GFPGANWrapper(
                        device=self.device,
                        weights_dir=self.weights_dir
                    )
                    logger.info("GFPGAN model loaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to load GFPGAN model: {e}")
                    raise
            return self._gfpgan

    def is_loaded(self, model_name='all'):
        if model_name == 'all':
            return self._realesrgan is not None and self._gfpgan is not None
        elif model_name.lower() == 'realesrgan':
            return self._realesrgan is not None
        elif model_name.lower() == 'gfpgan':
            return self._gfpgan is not None
        return False

    def unload_realesrgan(self):
        with self._realesrgan_lock:
            if self._realesrgan is not None:
                del self._realesrgan
                self._realesrgan = None
        gc.collect()

    def unload_gfpgan(self):
        with self._gfpgan_lock:
            if self._gfpgan is not None:
                del self._gfpgan
                self._gfpgan = None
        gc.collect()

    def cleanup(self):
        logger.info("Cleaning up models...")
        self.unload_realesrgan()
        self.unload_gfpgan()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Cleanup complete.")
