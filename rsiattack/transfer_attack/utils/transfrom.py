import random

import torch
import torch.nn.functional as F
import torch_dct as dct
import torchvision.transforms as transforms


def vertical_shift(x):
    w = x.shape[-2]
    step = random.randint(0, w)
    # step = np.random.randint(low = 0, high=w, dtype=np.int64)
    return x.roll(step, dims=-2)


def horizontal_shift(x):
    h = x.shape[-1]
    step = random.randint(0, h)
    # step = np.random.randint(low = 0, high=h, dtype=np.int64)
    return x.roll(step, dims=-1)


def vertical_flip(x):
    return x.flip(dims=(-2,))


def horizontal_flip(x):
    return x.flip(dims=(-1,))


def rotate180(x):
    return x.rot90(k=2, dims=(-2, -1))


def scale(x):
    return torch.rand(1)[0] * x


def resize(x):
    """
    Resize the input
    """
    w = x.shape[-2]
    h = x.shape[-1]
    scale_factor = random.uniform(0.6, 0.9)
    # scale_factor = 0.8
    new_h = int(h * scale_factor) + 1
    new_w = int(w * scale_factor) + 1
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    x = F.interpolate(x, size=(w, h), mode="bilinear", align_corners=False).clamp(0, 1)
    return x


def add_noise(x):
    return torch.clip(x + torch.zeros_like(x).uniform_(-16 / 255, 16 / 255), 0, 1)


def drop_out(x):
    return F.dropout2d(x, p=0.1, training=True)


def vertical_shift_sm(x):
    w = x.shape[-2]
    step = random.randint(0, 100)
    # step = np.random.randint(low = 0, high=w, dtype=np.int64)
    return x.roll(step, dims=-2)


def horizontal_shift_sm(x):
    h = x.shape[-1]
    step = random.randint(0, 100)
    # step = np.random.randint(low = 0, high=h, dtype=np.int64)
    return x.roll(step, dims=-1)


def dct_tran(x):
    """
    Discrete Fourier Transform
    """
    dctx = dct.dct_2d(x)  # torch.fft.fft2(x, dim=(-2, -1))
    _, _, w, h = dctx.shape
    low_ratio = 0.4
    low_w = int(w * low_ratio)
    low_h = int(h * low_ratio)
    # dctx[:, :, -low_w:, -low_h:] = 0
    dctx[:, :, -low_w:, :] = 0
    dctx[:, :, :, -low_h:] = 0
    dctx = dctx  # * self.mask.reshape(1, 1, w, h)
    idctx = dct.idct_2d(dctx)
    return idctx


Blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
solarizer = transforms.RandomSolarize(threshold=0.5)


def Solarize(x):

    solarized_imgs = [solarizer(x) for _ in range(4)]
    x = solarized_imgs[0] + x
    return x


def AdjustSharpness(x):
    sharpness = transforms.RandomAdjustSharpness(sharpness_factor=10, p=0.5)
    x = sharpness(x)
    return x


def Invert(x):
    invert = transforms.RandomInvert(p=1)

    x = invert(x)
    return x


def Jitter(x):
    jitter = transforms.ColorJitter(brightness=1.5, contrast=3)
    x = jitter(x)
    return x


op = [
    vertical_shift,
    horizontal_shift,
    vertical_flip,
    horizontal_flip,
    rotate180,
    scale,
    resize,
    add_noise,
    drop_out,
    Jitter,
]

op_bk = [
    vertical_shift,
    horizontal_shift,
    vertical_flip,
    horizontal_flip,
    rotate180,
    scale,
    resize,
    add_noise,
    drop_out,
    Solarize,
    Invert,
]
