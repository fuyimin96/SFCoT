'''
Description: Patch-wise Attack for Fooling Deep Neural Network
http://arxiv.org/abs/2007.06765
'''
from rsiattack import ATTACK
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

class PI_FGSM(ATTACK):
    def __init__(self, parser):
        self.loss = nn.CrossEntropyLoss()
        # Please do not modify the code under any circumstances.
        parser = self.create_options(parser)
        self.args = self.add_options(parser)
        
        super(PI_FGSM,self).__init__(self.args)
        self.amplification = 10.0
        
    
    def add_options(self, parser):
        parser.add_argument('--alpha', type=float, default=1)
        parser.add_argument('--eps', type=float, default=16)
        parser.add_argument('--epochs', type=int, default=100)
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
        alpha = epsilon / args.epochs
        alpha_beta = alpha * self.amplification
        gamma = 16.0
        for images, te, filename in tqdm(data_loader):
            images_adv = images.clone().detach().to(device)
            gt = self.get_tar_labels(images)
            images = images.to(device)
            net = net.to(device)
            te = te.to(device)
            gt = gt.to(device)
            amplification = 0.0
            for _ in range(args.epochs):
                
                images_adv_ = images_adv.clone()
                images_adv_.requires_grad = True

                output = net(images_adv_)

                cost = self.loss(output, te).to(args.device)
                cost.backward()
                noise = images_adv_.grad
                amplification += alpha_beta * torch.sign(noise)
                cut_noise = torch.clamp(abs(amplification) - epsilon, 0, 10000.0) * torch.sign(amplification)
                projection = gamma * torch.sign(cut_noise)
                images_adv_ = images_adv_ + alpha_beta * torch.sign(noise) + projection
                images_adv_ = torch.where(images_adv_ > images + epsilon, 
                                        images + epsilon , images_adv_)
                images_adv_ = torch.where(images_adv_ < images - epsilon, 
                                        images - epsilon , images_adv_)
                images_adv = torch.clamp(images_adv_.detach(),0,1)

            self.save_images(images_adv, te, filename)
        
        eval_result = self.eval()

        return eval_result
    
    def clip_by_tensor(self, t, t_min, t_max):
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result