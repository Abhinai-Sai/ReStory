"""Application configuration."""
import os
import torch


class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
    MODEL_DIR = os.path.join(BASE_DIR, 'weights')
    LOG_FILE = os.path.join(BASE_DIR, 'logs', 'app.log')

    MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Render Free Tier Resource Strategy Settings
    MAX_INPUT_DIMENSION = 256
    ESRGAN_SCALE = 2
    ESRGAN_TILE = 128
    TILE_PAD = 10
    HALF = False
    WORKERS = 1
    THREADS = 1
    MAX_CONCURRENT_RESTORATIONS = 1

    SECRET_KEY = os.environ.get('SECRET_KEY', 'default-secret-key-change-in-production')
