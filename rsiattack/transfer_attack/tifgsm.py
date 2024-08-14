'''
Description: Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks
https://arxiv.org/abs/1904.02884
'''
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as st
from torch.utils.data import DataLoader
from tqdm import tqdm

from rsiattack import ATTACK


class TI_FGSM(ATTACK):
    def __init__(self, parser):
        self.loss = nn.CrossEntropyLoss()
        # Please do not modify the code under any circumstances.
        parser = self.create_options(parser)
        self.args = self.add_options(parser)
        
        super(TI_FGSM,self).__init__(self.args)
        
    
    def add_options(self, parser):
        parser.add_argument('--alpha', type=float, default=1)
        parser.add_argument('--eps', type=float, default=16)
        parser.add_argument('--epochs', type=int, default=30)
        parser.add_argument('--TI2p', type=float, default=0.7,
                            help='The probability of undergoing transformations in TI-FGSM.')
        parser.add_argument('--len_kernel', type=int, default=15)
        parser.add_argument('--nsig', type=int, default=3)
        parser.add_argument('--kernel_name', type=str, default='gaussian')
        args = parser.parse_args()
        args.att = self.__class__.__name__
        return args

    def attack(self):  
        args = self.args
        device = args.device
        net = self.net.to(device)
        warnings.filterwarnings("ignore", category=UserWarning)
        data_loader = DataLoader(self.dataset, batch_size=args.batch_size, 
                                 num_workers=10)
        stacked_kernel = torch.from_numpy(kernel_generation(args)).to(args.device)
        epsilon = args.eps / 255
        alpha = args.alpha / 255
        for images, te, filename in tqdm(data_loader):
            images_adv = images.clone().detach().to(device)
            gt = self.get_tar_labels(images)
            images = images.to(device)
            net = net.to(device)
            te = te.to(device)
            gt = gt.to(device)
            for _ in range(args.epochs):
                
                images_adv_ = images_adv.clone()
                images_adv_.requires_grad = True

                output = net(images_adv_)

                cost = - self.loss(output, te).to(args.device)
                
                grad = torch.autograd.grad(cost, images_adv_, retain_graph=False, create_graph=False)[0]
                if torch.rand(1) < args.TI2p:
                    grad = F.conv2d(grad, stacked_kernel, stride=1, padding="same", groups=3)
                grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)

                images_adv_ = images_adv_ - alpha * torch.sign(grad)
                images_adv_ = torch.where(images_adv_ > images + epsilon, 
                                        images + epsilon , images_adv_)
                images_adv_ = torch.where(images_adv_ < images - epsilon, 
                                        images - epsilon , images_adv_)
                images_adv = torch.clamp(images_adv_.detach(),0,1)

            self.save_images(images_adv, te, filename)
        
        eval_result = self.eval()

        return eval_result
    

def kernel_generation(args):
    if args.kernel_name == "gaussian":
        kernel = gkern(args.len_kernel, args.nsig).astype(np.float32)
    elif args.kernel_name == "linear":
        kernel = lkern(args.len_kernel).astype(np.float32)
    elif args.kernel_name == "uniform":
        kernel = ukern(args.len_kernel).astype(np.float32)
    else:
        raise NotImplementedError

    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)
    return stack_kernel

def gkern(kernlen=15, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel



def ukern(kernlen=15):
    kernel = np.ones((kernlen, kernlen)) * 1.0 / (kernlen * kernlen)
    return kernel

def lkern(kernlen=15):
    kern1d = 1 - np.abs(
        np.linspace((-kernlen + 1) / 2, (kernlen - 1) / 2, kernlen)
        / (kernlen + 1)
        * 2
    )
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel