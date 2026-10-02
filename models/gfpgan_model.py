import os
import logging
import torch
import urllib.request
from gfpgan import GFPGANer

logger = logging.getLogger(__name__)

class GFPGANWrapper:
    def __init__(self, device=None, weights_dir='weights'):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights_dir = weights_dir
        self.model_name = 'GFPGANv1.4'
        self.weights_path = os.path.join(self.weights_dir, f'{self.model_name}.pth')
        self.model_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'
        
        self.gfpganer = None
        self._load_model()

    def download_weights(self):
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
            
        if not os.path.exists(self.weights_path):
            logger.info(f"Downloading {self.model_name} weights to {self.weights_path}...")
            try:
                urllib.request.urlretrieve(self.model_url, self.weights_path)
                logger.info("Download completed.")
            except Exception as e:
                logger.error(f"Failed to download weights: {e}")
                raise
        else:
            logger.info(f"Weights already exist at {self.weights_path}")

    def _load_model(self):
        self.download_weights()
        
        logger.info("Initializing GFPGANer...")
        try:
            self.gfpganer = GFPGANer(
                model_path=self.weights_path,
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device
            )
        except Exception as e:
            logger.error(f"Error initializing GFPGANer: {e}")
            raise

    @torch.inference_mode()
    def enhance_faces(self, img_bgr, realesrgan_model=None):
        if self.gfpganer is None:
            raise RuntimeError("GFPGAN model is not loaded.")
            
        logger.info("Enhancing faces with GFPGAN...")
        
        # Set bg_upsampler to None since background was already enhanced by Real-ESRGAN
        self.gfpganer.bg_upsampler = None
        
        try:
            cropped_faces, restored_faces, restored_img = self.gfpganer.enhance(
                img_bgr,
                has_aligned=False,
                only_center_face=True,
                paste_back=True,
                weight=0.5
            )
            
            has_faces = restored_faces is not None and len(restored_faces) > 0
            if not has_faces:
                logger.info("No faces detected in the image.")
                return img_bgr, False
                
            return restored_img, True
            
        except Exception as e:
            logger.error(f"Failed during face enhancement: {e}")
            raise
