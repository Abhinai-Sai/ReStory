"""Minimal StyleGAN2 arch stubs for GFPGAN."""
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


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, demodulate=True, upsample=False, downsample=False, blur_kernel=[1,3,3,1], **kwargs):
        super().__init__()
        self.eps = 1e-8
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.downsample = downsample
        self.demodulate = demodulate
        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

    def forward(self, x, style):
        b, c, h, w = x.shape
        style = self.modulation(style).view(b, 1, c, 1, 1)
        weight = self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(b, self.out_channel, 1, 1, 1)
        weight = weight.view(b * self.out_channel, c, self.kernel_size, self.kernel_size)
        x = x.view(1, b * c, h, w)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            _, _, h, w = x.shape
        p = self.kernel_size // 2
        x = F.pad(x, [p, p, p, p])
        out = F.conv2d(x, weight, groups=b)
        _, _, h, w = out.shape
        out = out.view(b, self.out_channel, h, w)
        return out

class StyleConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, upsample=False, blur_kernel=[1,3,3,1], demodulate=True, **kwargs):
        super().__init__()
        self.conv = ModulatedConv2d(in_channel, out_channel, kernel_size, style_dim, upsample=upsample, blur_kernel=blur_kernel, demodulate=demodulate)
        self.activate = ScaledLeakyReLU(0.2)

    def forward(self, x, style, **kwargs):
        out = self.conv(x, style)
        return self.activate(out)

class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1,3,3,1]):
        super().__init__()
        self.upsample = upsample
        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, style, skip=None):
        out = self.conv(x, style)
        out = out + self.bias
        if skip is not None:
            if self.upsample:
                skip = F.interpolate(skip, scale_factor=2, mode='bilinear', align_corners=False)
            out = out + skip
        return out

class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, x):
        batch = x.shape[0]
        return self.input.repeat(batch, 1, 1, 1)

class StyleGAN2Generator(nn.Module):
    def __init__(self, out_size, num_style_feat=512, num_mlp=8, channel_multiplier=2, narrow=1, **kwargs):
        super().__init__()
        self.out_size = out_size
        self.num_style_feat = num_style_feat
        # Simplified - just enough to load weights
        style_mlp_layers = [nn.Linear(num_style_feat, num_style_feat)]
        for _ in range(num_mlp - 1):
            style_mlp_layers.append(nn.Linear(num_style_feat, num_style_feat))
        self.style_mlp = nn.Sequential(*style_mlp_layers)

    def forward(self, styles, **kwargs):
        return styles
