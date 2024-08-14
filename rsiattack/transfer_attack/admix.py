'''
Description: Admix: Enhancing the Transferability of Adversarial Attacks
https://arxiv.org/abs/2102.00436
'''
from rsiattack import ATTACK
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

class ADMIX(ATTACK):
    def __init__(self, parser):
        self.loss = nn.CrossEntropyLoss()
        # Please do not modify the code under any circumstances.
        parser = self.create_options(parser)
        self.args = self.add_options(parser)
        
        super(ADMIX,self).__init__(self.args)
        
    
    def add_options(self, parser):
        parser.add_argument('--alpha', type=float, default=1)
        parser.add_argument('--eps', type=float, default=16)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--mu', type=float, default=1.0)
        parser.add_argument('--admix_strength', type=float, default=0.2)
        parser.add_argument('--num_scale', type=int, default=5)
        parser.add_argument('--num_admix', type=int, default=3)
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
            last_g = torch.zeros_like(images, dtype=torch.float32)
            for _ in range(args.epochs):
                
                images_adv_ = images_adv.clone()
                images_adv_.requires_grad = True

                output = net(self.transform(images_adv_))

                cost = - self.loss(output, te.repeat(args.num_scale*args.num_admix)).to(args.device)
                cost.backward()
                
                g = last_g * args.mu + images_adv_.grad / \
                                        torch.norm(images_adv_.grad, p=1)
                images_adv_ = images_adv_ - alpha * torch.sign(g)
                images_adv_ = torch.where(images_adv_ > images + epsilon, 
                                        images + epsilon, images_adv_)
                images_adv_ = torch.where(images_adv_ < images - epsilon, 
                                        images - epsilon, images_adv_)
                images_adv = torch.clamp(images_adv_.detach(), 0, 1)
                last_g = g

            self.save_images(images_adv, te, filename)
        end = time.time()
        self.spend_time = start - end
        eval_result = self.eval()

        return eval_result
    
    def transform(self, x):
        """
        Admix the input for Admix Attack
        """
        admix_images = torch.concat([(x + self.args.admix_strength * x[torch.randperm(x.size(0))].detach()) for _ in range(self.args.num_admix)], dim=0)
        return torch.concat([admix_images / (2 ** i) for i in range(self.args.num_scale)])