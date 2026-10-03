"""Minimal download utility shim."""
import os
import urllib.request

def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    if model_dir is None:
        from torch.hub import get_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)
    if file_name is None:
        parts = url.split('/')
        file_name = parts[-1]
    cached_file = os.path.join(model_dir, file_name)
    if not os.path.exists(cached_file):
        print(f"Downloading: {url} to {cached_file}")
        urllib.request.urlretrieve(url, cached_file)
    return cached_file
