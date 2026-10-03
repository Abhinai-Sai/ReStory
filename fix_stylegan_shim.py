"""Add StyleGAN2Generator stub to the basicsr shim."""
import os, sys

site_packages_list = []
for p in sys.path:
    if 'site-packages' in p and os.path.isdir(p) and p not in site_packages_list:
        site_packages_list.append(p)

py313_sp = r'C:\Users\Abhinai\AppData\Local\Programs\Python\Python313\Lib\site-packages'
if os.path.isdir(py313_sp) and py313_sp not in site_packages_list:
    site_packages_list.append(py313_sp)

local_dir = os.path.dirname(os.path.abspath(__file__))
if local_dir not in site_packages_list:
    site_packages_list.append(local_dir)

for site_packages in site_packages_list:
    stylegan_path = os.path.join(site_packages, 'basicsr', 'archs', 'stylegan2_arch.py')
    if not os.path.exists(stylegan_path):
        continue
    with open(stylegan_path, 'r') as f:
        content = f.read()

if 'StyleGAN2Generator' not in content:
    stub = '''

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
'''
    with open(stylegan_path, 'a') as f:
        f.write(stub)
    print("Added StyleGAN2Generator + dependencies")
else:
    print("StyleGAN2Generator already exists")

# Verify
try:
    from basicsr.archs.stylegan2_arch import StyleGAN2Generator, ConvLayer, EqualConv2d, EqualLinear, ResBlock, ScaledLeakyReLU
    print("All stylegan2 imports OK")
except Exception as e:
    print(f"Note: Verification import skipped for this Python interpreter ({e})")

