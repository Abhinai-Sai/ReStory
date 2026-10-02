"""Create a minimal basicsr shim so that realesrgan/gfpgan can import."""
import os
import sys

site_packages = None
for p in sys.path:
    if 'site-packages' in p and os.path.isdir(p):
        site_packages = p
        break

if not site_packages:
    print("ERROR: Could not find site-packages directory")
    sys.exit(1)

basicsr_dir = os.path.join(site_packages, 'basicsr')
print(f"Creating basicsr shim at: {basicsr_dir}")

dirs_to_create = [
    basicsr_dir,
    os.path.join(basicsr_dir, 'utils'),
    os.path.join(basicsr_dir, 'archs'),
    os.path.join(basicsr_dir, 'ops'),
    os.path.join(basicsr_dir, 'ops', 'fused_act'),
    os.path.join(basicsr_dir, 'data'),
    os.path.join(basicsr_dir, 'losses'),
    os.path.join(basicsr_dir, 'models'),
    os.path.join(basicsr_dir, 'metrics'),
]

for d in dirs_to_create:
    os.makedirs(d, exist_ok=True)

# basicsr/__init__.py
with open(os.path.join(basicsr_dir, '__init__.py'), 'w') as f:
    f.write('"""Minimal basicsr shim for realesrgan/gfpgan inference."""\n')

# basicsr/utils/__init__.py
with open(os.path.join(basicsr_dir, 'utils', '__init__.py'), 'w') as f:
    f.write('''"""Minimal basicsr.utils shim."""
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
''')

# basicsr/utils/registry.py
with open(os.path.join(basicsr_dir, 'utils', 'registry.py'), 'w') as f:
    f.write('''"""Minimal registry shim."""

class Registry:
    def __init__(self, name):
        self.name = name
        self._obj_map = {}

    def register(self, obj=None, name=None):
        if obj is None:
            def wrapper(fn_or_class):
                n = name or fn_or_class.__name__
                self._obj_map[n] = fn_or_class
                return fn_or_class
            return wrapper
        n = name or obj.__name__
        self._obj_map[n] = obj
        return obj

    def get(self, name):
        return self._obj_map.get(name)

    def __contains__(self, name):
        return name in self._obj_map

ARCH_REGISTRY = Registry('arch')
MODEL_REGISTRY = Registry('model')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')
DATASET_REGISTRY = Registry('dataset')
''')

# basicsr/utils/download_util.py
with open(os.path.join(basicsr_dir, 'utils', 'download_util.py'), 'w') as f:
    f.write('''"""Minimal download utility shim."""
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
''')

# basicsr/utils/img_process_util.py
with open(os.path.join(basicsr_dir, 'utils', 'img_process_util.py'), 'w') as f:
    f.write('''"""Minimal image process util shim."""
import torch
import torch.nn.functional as F

def filter2D(img, kernel):
    b, c, h, w = img.shape
    k = kernel.shape[-1]
    p = k // 2
    img = F.pad(img, (p, p, p, p), mode="reflect")
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(c, 1, 1, 1)
    return F.conv2d(img, kernel, groups=c)
''')

# basicsr/archs/__init__.py
with open(os.path.join(basicsr_dir, 'archs', '__init__.py'), 'w') as f:
    f.write('''"""Minimal archs shim."""
def build_network(opt):
    raise NotImplementedError("build_network not available in shim")
''')

# basicsr/archs/arch_util.py
with open(os.path.join(basicsr_dir, 'archs', 'arch_util.py'), 'w') as f:
    f.write('''"""Minimal arch_util shim."""
import torch.nn as nn

def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
''')

# basicsr/archs/stylegan2_arch.py (needed by gfpgan)
with open(os.path.join(basicsr_dir, 'archs', 'stylegan2_arch.py'), 'w') as f:
    f.write('''"""Minimal StyleGAN2 arch stubs for GFPGAN."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return F.leaky_relu(x, negative_slope=self.negative_slope) * math.sqrt(2)

class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        self.bias = nn.Parameter(torch.zeros(out_channel)) if bias else None

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init)) if bias else None
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        self.activation = activation

    def forward(self, x):
        out = F.linear(x, self.weight * self.scale, bias=self.bias * self.lr_mul if self.bias is not None else None)
        if self.activation == 'fused_lrelu':
            out = F.leaky_relu(out, negative_slope=0.2) * math.sqrt(2)
        return out

class ConvLayer(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, downsample=False, blur_kernel=[1,3,3,1], bias=True, activate=True, pad=None):
        layers = []
        if downsample:
            stride = 2
            p = (kernel_size - 1) // 2
        else:
            stride = 1
            p = kernel_size // 2 if pad is None else pad
        layers.append(EqualConv2d(in_channel, out_channel, kernel_size, padding=p, stride=stride, bias=bias and not activate))
        if activate:
            layers.append(ScaledLeakyReLU(0.2))
        super().__init__(*layers)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1,3,3,1]):
        super().__init__()
        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)
        self.skip = ConvLayer(in_channel, out_channel, 1, downsample=True, activate=False, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x)
        return (out + skip) / math.sqrt(2)
''')

# basicsr/ops/__init__.py
with open(os.path.join(basicsr_dir, 'ops', '__init__.py'), 'w') as f:
    f.write('')

# basicsr/ops/fused_act/__init__.py
with open(os.path.join(basicsr_dir, 'ops', 'fused_act', '__init__.py'), 'w') as f:
    f.write('''"""Fused activation shim using pure PyTorch."""
import torch
import torch.nn as nn
import torch.nn.functional as F

def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2**0.5):
    if bias is not None:
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return F.leaky_relu(input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope) * scale
    return F.leaky_relu(input, negative_slope=negative_slope) * scale

class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2**0.5):
        super().__init__()
        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))
        else:
            self.bias = None
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, x):
        return fused_leaky_relu(x, self.bias, self.negative_slope, self.scale)
''')

# basicsr/data/__init__.py
with open(os.path.join(basicsr_dir, 'data', '__init__.py'), 'w') as f:
    f.write('')

# basicsr/data/data_util.py
with open(os.path.join(basicsr_dir, 'data', 'data_util.py'), 'w') as f:
    f.write('''def paths_from_folder(folder): return []
def paired_paths_from_folder(folders, keys, filename_tmpl): return []
def paired_paths_from_lmdb(folders, keys): return []
''')

# basicsr/data/degradations.py
with open(os.path.join(basicsr_dir, 'data', 'degradations.py'), 'w') as f:
    f.write('''def circular_lowpass_kernel(*args, **kwargs): return None
def random_mixed_kernels(*args, **kwargs): return None
def random_add_gaussian_noise_pt(*args, **kwargs): return args[0]
def random_add_poisson_noise_pt(*args, **kwargs): return args[0]
''')

# basicsr/data/transforms.py
with open(os.path.join(basicsr_dir, 'data', 'transforms.py'), 'w') as f:
    f.write('''def augment(*args, **kwargs): return args[0] if args else None
def paired_random_crop(*args, **kwargs): return args[0] if args else None
''')

# basicsr/losses/__init__.py
with open(os.path.join(basicsr_dir, 'losses', '__init__.py'), 'w') as f:
    f.write('''def build_loss(opt): raise NotImplementedError("Not available in shim")
''')

# basicsr/losses/gan_loss.py
with open(os.path.join(basicsr_dir, 'losses', 'gan_loss.py'), 'w') as f:
    f.write('''def r1_penalty(*args, **kwargs): return 0
''')

# basicsr/models/__init__.py
with open(os.path.join(basicsr_dir, 'models', '__init__.py'), 'w') as f:
    f.write('')

# basicsr/models/base_model.py
with open(os.path.join(basicsr_dir, 'models', 'base_model.py'), 'w') as f:
    f.write('''class BaseModel: pass
''')

# basicsr/models/srgan_model.py / sr_model.py
for name in ['srgan_model.py', 'sr_model.py']:
    with open(os.path.join(basicsr_dir, 'models', name), 'w') as f:
        f.write(f'''class {"SRGANModel" if "srgan" in name else "SRModel"}: pass
''')

# basicsr/metrics/__init__.py
with open(os.path.join(basicsr_dir, 'metrics', '__init__.py'), 'w') as f:
    f.write('''def calculate_metric(*args, **kwargs): return {}
''')

# basicsr/archs/rrdbnet_arch.py (the real one we have locally)
rrdbnet_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'rrdbnet_arch.py')
if os.path.exists(rrdbnet_src):
    import shutil
    shutil.copy2(rrdbnet_src, os.path.join(basicsr_dir, 'archs', 'rrdbnet_arch.py'))
    print("Copied rrdbnet_arch.py to basicsr shim")

print("basicsr shim created successfully!")

# Verify
import importlib
import basicsr
print(f"basicsr location: {basicsr.__file__}")
from basicsr.utils import scandir
print("scandir import: OK")
from basicsr.utils.registry import ARCH_REGISTRY
print("ARCH_REGISTRY import: OK")
from basicsr.utils.download_util import load_file_from_url
print("load_file_from_url import: OK")

