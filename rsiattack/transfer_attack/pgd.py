'''
Description: Towards Deep Learning Models Resistant to Adversarial Attacks
https://arxiv.org/abs/1706.06083
'''
from rsiattack import ATTACK
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class PGD(ATTACK):
    def __init__(self, parser):
        self.loss = nn.CrossEntropyLoss()
        # Please do not modify the code under any circumstances.
        parser = self.create_options(parser)
        self.args = self.add_options(parser)
        
        super(PGD,self).__init__(self.args)
        
    
    def add_options(self, parser):
        parser.add_argument('--alpha', type=float, default=1.0)
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
        for images, te, filename in tqdm(data_loader):
            gt = self.get_tar_labels(images)
            images = images.to(device)
            net = net.to(device)
            te = te.to(device)
            gt = gt.to(device)
            images_adv_ = images.detach().to(device)
            images_adv_ = images_adv_ + torch.empty_like(images_adv_).uniform_(-epsilon, epsilon)
            images_adv_ = torch.clamp(images_adv_, min=0, max=1).detach()
            for _ in range(args.epochs):
                
                images_adv_.requires_grad = True

                output = net(images_adv_)

                cost = self.loss(output, te).to(args.device)
                grad = - torch.autograd.grad(cost, images_adv_, 
                                           retain_graph=False, 
                                           create_graph=False)[0]
                images_adv_ = images_adv_.detach() - alpha * grad.sign()
                delta = torch.clamp(images_adv_ - images, min=-epsilon, max=epsilon)
                images_adv_ = torch.clamp(images + delta, min=0, max=1).detach()
            self.save_images(images_adv_, te, filename)
        eval_result = self.eval()
        return eval_result