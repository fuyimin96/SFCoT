'''
Description: Transferable Adversarial Attacks for Remote Sensing 
            Object Recognition via Spatial-Frequency Co-Transformation
'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable as V
import random
from rsiattack import op, op_bk, CAM, ATTACK
from pytorch_wavelets import DWTForward, DWTInverse
random.seed(121)

class SFCoT(ATTACK):
    def __init__(self, parser):
        self.loss = nn.CrossEntropyLoss()
        # Please do not modify the code under any circumstances.
        parser = self.create_options(parser)
        self.args = self.add_options(parser)
        
        super(SFCoT, self).__init__(self.args)
        self.num_copies = self.args.num_copies
        self.num_block = self.args.num_block
        self.op = op
        self.op_bk = op_bk
    
    def add_options(self, parser):
        parser.add_argument('--alpha', type=float, default=1)
        parser.add_argument('--eps', type=float, default=16)
        parser.add_argument('--epochs', type=int, default=30)
        parser.add_argument('--num_block', type=int, default=3)
        parser.add_argument('--num_copies', type=int, default=5)
        parser.add_argument('--mu', type=float, default=1.0)
        parser.add_argument('--th', type=float, default=0.5)
        parser.add_argument('--move_max', type=int, default=10)
        parser.add_argument('--resize_max', type=float, default=0.4)
        parser.add_argument("--rholl", type=float, default=0.1)
        parser.add_argument('--wave', type=str, default='db3')
        parser.add_argument('--no_sp', action='store_true')
        parser.add_argument('--no_fp', action='store_true')
        args = parser.parse_args()
        args.att = self.__class__.__name__
        return args

    def attack(self):
        args = self.args
        device = args.device
        net = self.net.to(device)
        device = args.device
        data_loader = DataLoader(self.dataset, batch_size=args.batch_size, 
                                 num_workers=10)
        for images, te, filename in tqdm(data_loader):
            images = images.to(device)
            net = net.to(device)
            te = te.to(device)
            trans = self.SFT(images, te, net)

            self.save_images(trans, te, filename)
        
        eval_result = self.eval()
        return eval_result
    
    def SFT(self, images, te, model):
        args = self.args
        cam_list = []
        if not self.args.no_sp:
            for image in images:
                image = image.to(args.device)
                cam = CAM(image, self.net, th=self.args.th)
                cam_list.append(cam)
        eps = args.eps / 255.0
        alpha = args.alpha / 255.0
        images_adv = images.clone()
        device = args.device
        last_g = torch.zeros_like(images).to(device)
        for _ in range(args.epochs):
            
            x_adv = images_adv.clone()
            x_adv = V(x_adv, requires_grad = True)
            if args.no_sp and args.no_fp:
                output_v3 = model(x_adv)
                loss = F.cross_entropy(output_v3, te).to(args.device)
            else:
                output_v3 = model(self.transform(x_adv, cam_list))
                loss = F.cross_entropy(output_v3, te.repeat(args.num_copies)).to(args.device)
                
            grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
            g = last_g * args.mu + grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            
            
            images_adv = images_adv + alpha * torch.sign(g)
            images_adv = torch.where(images_adv > images + eps, 
                                    images + eps, images_adv)
            images_adv = torch.where(images_adv < images - eps, 
                                    images - eps, images_adv)
            images_adv = torch.clamp(images_adv.detach(), 0, 1)
            last_g = g
        return images_adv.detach()

    def blocktransform(self, x, cam_list, choice=-1):
        x_copy = x.clone()
        if not self.args.no_fp:
            x_copy = self.dwt_trans(x_copy)
        if not self.args.no_sp:
            for bat, _ in enumerate(x_copy):
                x1, y1, x2, y2 = self.changeCAM(cam_list[bat])
                chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op_bk), dtype=np.int32)
                x_copy[bat, :, 0:x1, 0:y1] = self.op_bk[chosen](x_copy[bat, :, 0:x1, 0:y1].unsqueeze(0))
                chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op_bk), dtype=np.int32)
                x_copy[bat, :, x1:x2, 0:y1] = self.op_bk[chosen](x_copy[bat, :, x1:x2, 0:y1].unsqueeze(0))
                chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op_bk), dtype=np.int32)
                x_copy[bat, :, x2:223, 0:y1] = self.op_bk[chosen](x_copy[bat, :, x2:223, 0:y1].unsqueeze(0))
                
                chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op_bk), dtype=np.int32)
                x_copy[bat, :, 0:x1, y1:y2] = self.op_bk[chosen](x_copy[bat, :, 0:x1, y1:y2].unsqueeze(0))
                chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op), dtype=np.int32)
                x_copy[bat, :, x1:x2, y1:y2] = self.op[chosen](x_copy[bat, :, x1:x2, y1:y2].unsqueeze(0))
                chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op_bk), dtype=np.int32)
                x_copy[bat, :, x2:223, y1:y2] = self.op_bk[chosen](x_copy[bat, :, x2:223, y1:y2].unsqueeze(0))
                
                chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op_bk), dtype=np.int32)
                x_copy[bat, :, 0:x1, y2:223] = self.op_bk[chosen](x_copy[bat, :, 0:x1, y2:223].unsqueeze(0))
                chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op_bk), dtype=np.int32)
                x_copy[bat, :, x1:x2, y2:223] = self.op_bk[chosen](x_copy[bat, :, x1:x2, y2:223].unsqueeze(0))
                chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op_bk), dtype=np.int32)
                x_copy[bat, :, x2:223, y2:223] = self.op_bk[chosen](x_copy[bat, :, x2:223, y2:223].unsqueeze(0))
            
            
        return x_copy

    def block_trans(self, image, block):
        chosen_op = self.op[np.random.randint(0, high=len(self.op), dtype=np.int32)]
        x_start, x_end, y_start, y_end = block
        x_block = image[:, x_start:x_end, y_start:y_end]
        return chosen_op(x_block.unsqueeze(0)).squeeze(0)
    
    def dwt_trans(self, images):
        args = self.args
        rholl = args.rholl
        device = args.device
        xfm = DWTForward(J=1, wave=args.wave, mode='zero').to(device)
        Yl, Yh = xfm(images)
        maskl = torch.rand_like(Yl) * rholl * 2 + 1 - rholl
        Yl = Yl * maskl.to(device)
        ifm = DWTInverse(wave=args.wave, mode='zero').to(device)
        Y = ifm((Yl, Yh))
        
        return Y
        

    def transform(self, x, cam_list):
        """
        Scale the input for BlockShuffle
        """ 
        return torch.cat([self.blocktransform(x, cam_list) for _ in range(self.args.num_copies)])
        # return torch.cat([self.dwt_trans(x) for _ in range(self.args.num_copies)])
        
                
    def changeCAM(self, clist):
        x_move_pix = random.randint(-self.args.move_max, self.args.move_max)
        y_move_pix = random.randint(-self.args.move_max, self.args.move_max)
        x1, y1, x2, y2 = clist
        resize_rate = random.uniform(-self.args.resize_max, self.args.resize_max)
        x_resize_pix = int((x1 - x2) * resize_rate)
        y_resize_pix = int((y1 - y2) * resize_rate)
        x1_ = min(max(x1 - x_resize_pix + x_move_pix, 10), 213) 
        x2_ = max(min(x2 + x_resize_pix + x_move_pix, 213) ,10)
        y1_ = min(max(y1 - y_resize_pix + y_move_pix, 10) , 213)
        y2_ = max(min(y2 + y_resize_pix + y_move_pix, 213) , 10)
        if x1_ == x2_:
            x2_ = x2_ + 10
            x1_ = x1_ - 10
        if y1 == y2:
            y2_ = y2_ + 10
            y1_ = y1_ - 10
        return x1_, y1_, x2_ , y2_