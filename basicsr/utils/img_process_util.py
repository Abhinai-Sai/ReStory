"""Minimal image process util shim."""
import torch
import torch.nn.functional as F

def filter2D(img, kernel):
    b, c, h, w = img.shape
    k = kernel.shape[-1]
    p = k // 2
    img = F.pad(img, (p, p, p, p), mode="reflect")
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(c, 1, 1, 1)
    return F.conv2d(img, kernel, groups=c)
