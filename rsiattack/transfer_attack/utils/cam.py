import numpy as np
from PIL import Image
import PIL

import torch
from torch import nn
from torchvision import transforms
import torchvision.transforms.functional as TF

class Flatten(nn.Module):
    """One layer module that flattens its input."""
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)
    
def get_class_name(c):
    labels = np.loadtxt('stuff/synset_words.txt', str, delimiter='\t')
    return ' '.join(labels[c].split(',')[0].split()[1:])

def GradCAM(img, c, features_fn, classifier_fn):
    torch.set_grad_enabled(True)
    feats = features_fn(img)
    _, N, H, W = feats.size()
    out = classifier_fn(feats)
    c_score = out[0, c]
    grads = torch.autograd.grad(c_score, feats)
    w = grads[0][0].mean(-1).mean(-1)
    sal = torch.matmul(w, feats.view(N, H*W))
    sal = sal.view(H, W).cpu().detach().numpy()
    sal = np.maximum(sal, 0)
    return sal

data_transform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
    ])

def CAM(image, model, th = 0.3):
    # Split model in two parts
    arch = model.__class__.__name__
    if arch == 'ResNet':
        features_fn = nn.Sequential(*list(model.children())[:-2])
        classifier_fn = nn.Sequential(*(list(model.children())[-2:-1] + [Flatten()] + list(model.children())[-1:]))
    elif arch == 'VGG':
        features_fn = nn.Sequential(*list(model.features.children())[:-1])
        classifier_fn = nn.Sequential(*(list(model.features.children())[-1:] + [Flatten()] + list(model.classifier.children())))
    elif arch == 'DenseNet':
        features_fn = model.features
        classifier_fn = nn.Sequential(*([nn.AvgPool2d(7, 1), Flatten()] + [model.classifier]))
    elif arch == 'Inception_ResNetv2':
        features_fn = nn.Sequential(*(list(model.features.children()) + [model.conv]))
        classifier_fn = nn.Sequential(*([model.global_average_pooling, Flatten(), model.linear]))
    elif arch == 'InceptionV3':
        features_fn = nn.Sequential(*list(model.features.children())[:-1])
        classifier_fn = nn.Sequential(*(list(model.features.children())[-1:] + [Flatten()] + list(model.output.children())))
        
    model.eval()
    with torch.no_grad():
        c = torch.argmax(model(image.unsqueeze(0)), dim=1).squeeze(0)
        sal = GradCAM(image.unsqueeze(0), int(c), features_fn, classifier_fn)
        sal = Image.fromarray(sal)
        sal = sal.resize((224,224))
        tensor = torch.from_numpy(np.array(sal))
        
        th = (tensor.max() - tensor.min()) * th +  tensor.min()
        
        x1 = (tensor >= th).sum(1).nonzero()[0].item()
        x2 = (tensor >= th).sum(1).nonzero()[-1].item()
        y1 = (tensor >= th).sum(0).nonzero()[0].item()
        y2 = (tensor >= th).sum(0).nonzero()[-1].item()
        
        x1 = min(max(x1, 20) , 173)
        x2 = max(min(x2, 213), 50) 
        y1 = min(max(y1, 20) , 173)
        y2 = max(min(y2, 213), 50)
        if x1 == x2:
            x2 = x2 + 10
            x1 = x1 - 10
        if y1 == y2:
            y2 = y2 + 10
            y1 = y1 - 10
            
    
    return x1, y1, x2, y2