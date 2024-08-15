import os

import torch
import torchvision.models as models


def vgg16(num):
    net = models.vgg16(weights="IMAGENET1K_V1")
    num_ftrs = net.classifier[-1].in_features
    net.classifier[-1] = torch.nn.Linear(num_ftrs, num)
    return net


def vgg19(num):
    net = models.vgg19(weights="IMAGENET1K_V1")
    num_ftrs = net.classifier[-1].in_features
    net.classifier[-1] = torch.nn.Linear(num_ftrs, num)
    return net


def resnet34(num):
    net = models.resnet34(weights="IMAGENET1K_V1")
    num_ftrs = net.fc.in_features
    net.fc = torch.nn.Linear(num_ftrs, num)
    return net


def resnet50(num):
    net = models.resnet50(weights="IMAGENET1K_V1")
    num_ftrs = net.fc.in_features
    net.fc = torch.nn.Linear(num_ftrs, num)
    return net


def densenet121(num):
    net = models.densenet121(weights="IMAGENET1K_V1")
    num_ftrs = net.classifier.in_features
    net.classifier = torch.nn.Linear(num_ftrs, num)
    return net


def inception_resv2(num):
    from .inception_resnet_v2 import Inception_ResNetv2

    net = Inception_ResNetv2(classes=num)
    num_ftrs = net.linear.in_features
    net.fc = torch.nn.Linear(num_ftrs, num)
    return net


models_mapping = {
    "vgg16": vgg16,
    "vgg19": vgg19,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "densenet121": densenet121,
    "inception_resv2": inception_resv2,
}


def get_model_with_pretrain(madel_type, args):
    model_path = os.path.join(args.model_dir, madel_type)
    model_dict = torch.load(model_path, map_location="cpu")
    fc = models_mapping[madel_type]
    net = fc(args.num)
    net.load_state_dict(model_dict)
    return net
