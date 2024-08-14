'''
Description: Improving Transferability of Adversarial Examples with Input Diversity
http://arxiv.org/abs/1803.06978
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from rsiattack import ATTACK


class DI2_FGSM(ATTACK):
    def __init__(self, parser):
        self.loss = nn.CrossEntropyLoss()
        # Please do not modify the code under any circumstances.
        parser = self.create_options(parser)
        self.args = self.add_options(parser)
        
        super(DI2_FGSM,self).__init__(self.args)
        
    
    def add_options(self, parser):
        parser.add_argument('--alpha', type=float, default=1.0)
        parser.add_argument('--eps', type=float, default=16)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--resize_rate', type=float, default=1.1)
        parser.add_argument('--DI2p', type=float, default=0.5,
                            help='The probability of undergoing transformations in DI2-FGSM.')
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
            images_adv = images.clone().detach().to(device)
            gt = self.get_tar_labels(images)
            images = images.to(device)
            net = net.to(device)
            te = te.to(device)
            gt = gt.to(device)
            for _ in range(args.epochs):
                
                images_adv_ = images_adv.clone()
                images_adv_.requires_grad = True

                output = net(self.input_diversity(images_adv_))

                cost = - self.loss(output, te).to(args.device)
                cost.backward()
                
                images_adv_ = images_adv_ - alpha * images_adv_.grad.sign()
                images_adv_ = torch.where(images_adv_ > images + epsilon, 
                                        images + epsilon , images_adv_)
                images_adv_ = torch.where(images_adv_ < images - epsilon, 
                                        images - epsilon , images_adv_)
                images_adv = torch.clamp(images_adv_.detach(),0,1)

            self.save_images(images_adv, te, filename)
        
        eval_result = self.eval()

        return eval_result
    
    def input_diversity(self, x):
        # get the parameters from the arguments
        args = self.args
        img_size = x.shape[-1]
        img_resize = int(img_size * args.resize_rate)

        # if the resize rate is less than 1, swap the variables
        if args.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        # generate a random integer between the original image size and the resized image size
        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        # resize the input image, with bilinear interpolation, to the size of the random integer
        rescaled = F.interpolate(
            x, size=[rnd, rnd], mode="bilinear", align_corners=False
        )
        # calculate the remaining height and width of the rescaled image
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        # generate a random integer for the top and bottom padding
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        # generate a random integer for the left and right padding
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        # pad the rescaled image with the generated padding
        padded = F.pad(
            rescaled,
            [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()],
            value=0,
        )

        # return the padded image with a probability of 0.5
        return padded if torch.rand(1) < args.DI2p else x