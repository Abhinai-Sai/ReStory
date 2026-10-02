import os
import logging
import torch
import urllib.request
from gfpgan import GFPGANer

logger = logging.getLogger(__name__)

GFPGAN_URLS = [
    'https://huggingface.co/gfpgan/GFPGAN/resolve/main/GFPGANv1.4.pth',
    'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'
]


class GFPGANWrapper:
    def __init__(self, device=None, weights_dir='weights'):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights_dir = weights_dir
        self.model_name = 'GFPGANv1.4'
        self.weights_path = os.path.join(self.weights_dir, f'{self.model_name}.pth')

        self.gfpganer = None
        self._load_model()

    def download_weights(self):
        os.makedirs(self.weights_dir, exist_ok=True)

        if not os.path.exists(self.weights_path) or os.path.getsize(self.weights_path) < 1_000_000:
            logger.info(f"Downloading {self.model_name} weights to {self.weights_path}...")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            success = False
            last_err = None

            for url in GFPGAN_URLS:
                try:
                    logger.info(f"Attempting GFPGAN download from {url}...")
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=120) as response, open(self.weights_path, 'wb') as out_file:
                        data = response.read()
                        out_file.write(data)

                    file_size = os.path.getsize(self.weights_path)
                    logger.info(f"GFPGAN Download finished. Size: {file_size / 1024 / 1024:.1f} MB")
                    if file_size >= 1_000_000:
                        success = True
                        break
                    else:
                        if os.path.exists(self.weights_path):
                            os.remove(self.weights_path)
                except Exception as e:
                    logger.warning(f"Failed GFPGAN download from {url}: {e}")
                    last_err = e

            if not success:
                if os.path.exists(self.weights_path):
                    os.remove(self.weights_path)
                raise RuntimeError(f"Failed to download GFPGAN weights from all mirrors: {last_err}")
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
