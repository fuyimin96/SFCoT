'''
Description: Frequency Domain Model Augmentation for Adversarial Attack
https://arxiv.org/abs/2207.05382

'''
from rsiattack import ATTACK
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable as V
import torch.nn.functional as F
import numpy as np
import time

def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.fft(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    V = Vc.real * W_r - Vc.imag * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V

def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)

def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
    v = torch.fft.ifft(tmp)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape).real

class SSA(ATTACK):
    def __init__(self, parser):
        self.loss = nn.CrossEntropyLoss()
        # Please do not modify the code under any circumstances.
        parser = self.create_options(parser)
        self.args = self.add_options(parser)
        
        super(SSA,self).__init__(self.args)
        
    
    def add_options(self, parser):
        parser.add_argument('--alpha', type=float, default=1.0)
        parser.add_argument('--eps', type=float, default=16)
        parser.add_argument('--mu', type=float, default=1.0)
        parser.add_argument('--epochs', type=int, default=30)
        parser.add_argument("--N", type=int, default=5, help="The number of Spectrum Transformations")
        parser.add_argument("--rho", type=float, default=0.5, help="Tuning factor")
        parser.add_argument("--sigma", type=float, default=16.0, help="Std of random noise")
        args = parser.parse_args()
        args.att = self.__class__.__name__
        return args

    def attack(self):

        args = self.args
        device = args.device
        net = self.net.to(device)
        
        data_loader = DataLoader(self.dataset, batch_size=args.batch_size, 
                                 num_workers=10)
        start = time.time()
        for images, te, filename in tqdm(data_loader):
            images = images.to(device)
            net = net.to(device)
            te = te.to(device)
            images_adv = self.Spectrum_Simulation_Attack(images, te, net)

            self.save_images(images_adv, te, filename)
        end = time.time()
        self.spend_time = start - end
        eval_result = self.eval()

        return eval_result
    
    def Spectrum_Simulation_Attack(self, images, te, model):
        args = self.args
        eps = args.eps / 255.0
        alpha = args.alpha / 255.0
        images_adv = images.clone()
        rho = args.rho
        device = args.device
        N = args.N
        sigma = args.sigma
        last_g = torch.zeros_like(images).to(device)
        for i in range(args.epochs):
            noise = 0
            if torch.rand(1) < 0.5:
                for n in range(N):
                    x_idct = self.ssa_trans(images)
                    x_idct = V(x_idct, requires_grad = True)
                    output_v3 = model(x_idct)
                    loss = F.cross_entropy(output_v3, te)
                    loss.backward()
                    noise = noise + x_idct.grad.data
                noise = noise / N
            else :
                images_adv.requires_grad = True
                output_v3 = model(images_adv)
                loss = F.cross_entropy(output_v3, te)
                loss.backward()
                noise = images_adv.grad
                images_adv.requires_grad = False
            g = last_g * args.mu + noise / torch.norm(noise, p=1)
            images_adv = images_adv + alpha * torch.sign(g)
            images_adv = torch.where(images_adv > images + eps, 
                                    images + eps, images_adv)
            images_adv = torch.where(images_adv < images - eps, 
                                    images - eps, images_adv)
            images_adv = torch.clamp(images_adv.detach(), 0, 1)
            last_g = g
        return images_adv.detach()
    
    def ssa_trans(self, images):
        args = self.args
        rho = args.rho
        device = args.device
        sigma = args.sigma
        gauss = torch.randn(images.shape) * (sigma / 255)
        gauss = gauss.to(device)
        x_dct = dct_2d(images + gauss).to(device)
        mask = (torch.rand_like(images) * 2 * rho + 1 - rho).to(device)
        x_idct = idct_2d(x_dct * mask)
        return x_idct