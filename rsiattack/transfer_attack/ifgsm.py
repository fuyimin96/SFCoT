import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rsiattack import ATTACK


class I_FGSM(ATTACK):
    def __init__(self, parser):
        self.loss = nn.CrossEntropyLoss()
        # Please do not modify the code under any circumstances.
        parser = self.create_options(parser)
        self.args = self.add_options(parser)

        super(I_FGSM, self).__init__(self.args)

    def add_options(self, parser):
        parser.add_argument("--alpha", type=float, default=1)
        parser.add_argument("--eps", type=float, default=16)
        parser.add_argument("--epochs", type=int, default=100)

        args = parser.parse_args()
        args.att = self.__class__.__name__
        return args

    def attack(self):

        args = self.args
        device = args.device
        net = self.net.to(device)

        data_loader = DataLoader(
            self.dataset, batch_size=args.batch_size, num_workers=10
        )
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
                cost = -self.loss(output, te).to(args.device)
                cost.backward()

                images_adv_ = images_adv_ - alpha * images_adv_.grad.sign()
                images_adv_ = torch.where(
                    images_adv_ > images + epsilon, images + epsilon, images_adv_
                )
                images_adv_ = torch.where(
                    images_adv_ < images - epsilon, images - epsilon, images_adv_
                )
                images_adv = torch.clamp(images_adv_.detach(), 0, 1)

            self.save_images(images_adv, te, filename)

        eval_result = self.eval()

        return eval_result
