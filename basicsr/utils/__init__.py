"""Minimal basicsr.utils shim."""
import os
import glob
import torch
import numpy as np

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files."""
    if isinstance(suffix, str):
        suffix = [suffix]
    root = dir_path
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            for fname in sorted(filenames):
                if suffix is None or any(fname.endswith(s) for s in suffix):
                    if full_path:
                        yield os.path.join(dirpath, fname)
                    else:
                        yield fname
    else:
        for fname in sorted(os.listdir(root)):
            if os.path.isfile(os.path.join(root, fname)):
                if suffix is None or any(fname.endswith(s) for s in suffix):
                    if full_path:
                        yield os.path.join(root, fname)
                    else:
                        yield fname

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy image(s) to tensor(s)."""
    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = img[:, :, [2, 1, 0]]  # BGR to RGB
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        if not float32:
            img = img.half()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    return _totensor(imgs, bgr2rgb, float32)

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Tensor to numpy image."""
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')
    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(np.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
        elif n_dim == 3:
            img_np = _tensor.numpy().transpose(1, 2, 0)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise ValueError(f'Only 2D, 3D, 4D tensors supported. Got {n_dim}D.')
        if rgb2bgr and img_np.ndim == 3 and img_np.shape[2] == 3:
            img_np = img_np[:, :, [2, 1, 0]]
        img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    return result[0] if len(result) == 1 else result

def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes."""
    import cv2
    img_array = np.frombuffer(content, dtype=np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_array, imread_flags.get(flag, cv2.IMREAD_UNCHANGED))
    if float32:
        img = img.astype(np.float32) / 255.
    return img

def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file."""
    import cv2
    if auto_mkdir:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    cv2.imwrite(file_path, img, params)

def get_root_logger(logger_name='basicsr', log_level=20, log_file=None):
    """Get root logger."""
    import logging
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        logger.setLevel(log_level)
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        logger.addHandler(handler)
    return logger

class FileClient:
    def __init__(self, backend='disk', **kwargs):
        self.backend = backend
    def get(self, filepath, client_key='default'):
        with open(filepath, 'rb') as f:
            return f.read()

class DiffJPEG:
    pass

class USMSharp:
    pass
