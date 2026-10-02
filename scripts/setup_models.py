import os
import sys
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS = {
    'RealESRGAN_x4plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    'GFPGANv1.4.pth': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'
}

def report_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = downloaded * 100 / total_size
    if percent > 100:
        percent = 100
    sys.stdout.write(f"\rProgress: {percent:.2f}% ({downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)")
    sys.stdout.flush()

def download_models(weights_dir='weights'):
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
        logger.info(f"Created directory: {weights_dir}")

    for filename, url in MODELS.items():
        filepath = os.path.join(weights_dir, filename)
        
        if os.path.exists(filepath):
            logger.info(f"\n{filename} already exists at {filepath}. Skipping download.")
            continue
            
        logger.info(f"\nDownloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath, reporthook=report_progress)
            sys.stdout.write("\n")
            logger.info(f"Successfully downloaded {filename}.")
            
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            if size_mb < 50:
                logger.warning(f"Downloaded file {filename} seems too small ({size_mb:.2f} MB). Might be corrupted.")
            else:
                logger.info(f"Verified file size: {size_mb:.2f} MB")
                
        except Exception as e:
            logger.error(f"\nFailed to download {filename}: {e}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    weights_directory = os.path.join(project_root, 'weights')
    
    download_models(weights_directory)
