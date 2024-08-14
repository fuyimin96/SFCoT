'''
Description: Boosting adversarial transferability by block shuffle and rotation
https://arxiv.org/abs/2308.10299
'''
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from rsiattack import ATTACK


class BSR(ATTACK):
    def __init__(self, parser):
        self.loss = nn.CrossEntropyLoss()
        # Please do not modify the code under any circumstances.
        parser = self.create_options(parser)
        self.args = self.add_options(parser)
        self.num_scale = self.args.num_scale
        self.num_block = self.args.num_block
        
        super(BSR,self).__init__(self.args)
        
    
    def add_options(self, parser):
        parser.add_argument('--alpha', type=float, default=1.0)
        parser.add_argument('--eps', type=float, default=16)
        parser.add_argument('--epochs', type=int, default=30)
        parser.add_argument('--mu', type=float, default=1.0)
        parser.add_argument('--num_scale', type=int, default=5)
        parser.add_argument('--num_block', type=int, default=3)
        args = parser.parse_args()
        args.att = self.__class__.__name__
        return args

    def attack(self):

        args = self.args
        device = args.device
        net = self.net.to(device)
        
        data_loader = DataLoader(self.dataset, batch_size=args.batch_size, 
                                 num_workers=10)
        epsilon = args.eps / 255
        alpha = args.alpha / 255
        start = time.time()
        for images, te, filename in tqdm(data_loader):
            images_adv = images.clone().detach().to(device)
            gt = self.get_tar_labels(images)
            images = images.to(device)
            net = net.to(device)
            te = te.to(device)
            gt = gt.to(device)
            last_g = torch.zeros_like(images).to(device)
            for _ in range(args.epochs):
                
                images_adv_ = images_adv.clone()
                images_adv_.requires_grad = True

                output = net(self.transform(images_adv_))

                cost = - self.get_loss(output, te).to(args.device)
                cost.backward()
                last_g = last_g * args.mu + images_adv_.grad / \
                                        torch.norm(images_adv_.grad, p=1)
                images_adv_ = images_adv_ - alpha * torch.sign(last_g)
                images_adv_ = torch.where(images_adv_ > images + epsilon, 
                                        images + epsilon , images_adv_)
                images_adv_ = torch.where(images_adv_ < images - epsilon, 
                                        images - epsilon , images_adv_)
                images_adv = torch.clamp(images_adv_.detach(),0,1)

            self.save_images(images_adv, te, filename)
        end = time.time()
        self.spend_time = start - end
        eval_result = self.eval()

        return eval_result
    
    def get_length(self, length):
        rand = np.random.uniform(size=self.num_block)
        rand_norm = np.round(rand/rand.sum()*length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
        return tuple(rand_norm)

    def shuffle_single_dim(self, x, dim):
        lengths = self.get_length(x.size(dim))
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips


    def shuffle(self, x):
        dims = [2,3]
        random.shuffle(dims)
        x_strips = self.shuffle_single_dim(x, dims[0])
        return torch.cat([torch.cat(self.shuffle_single_dim(x_strip, dim=dims[1]), dim=dims[1]) for x_strip in x_strips], dim=dims[0])

    def transform(self, x, **kwargs):
        """
        Scale the input for BSR
        """
        return torch.cat([self.shuffle(x) for _ in range(self.num_scale)])

    def get_loss(self, logits, label):
        """
        Calculate the loss
        """
        return self.loss(logits, label.repeat(self.num_scale))