#!/usr/bin/env python3
# code partially taken from https://github.com/waleedgondal/Texture-based-Super-Resolution-Network

import os
import argparse

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor
from torchvision.models import vgg19

class VGG(nn.Module):

    def __init__(self, layers=(0)):
        super(VGG, self).__init__()
        self.layers = layers
        self.model = vgg19(pretrained=True).features
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for name, layer in enumerate(self.model):
            x = layer(x)
            if name in self.layers:
                features.append(x)
                if len(features) == len(self.layers):
                    break
        return features

def distance(im1, im2, cuda=False):
    vgg_layers = [int(i) for i in opt.texture_layers]
    vgg_texture = VGG(layers=vgg_layers)
    if cuda:
        vgg_texture = vgg_texture.cuda()

    def gram_matrix(y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / ch
        return gram

    def criterion(a, b):
        return torch.mean(torch.abs((a-b)**2).view(-1))

    text_loss = []
    vgg1 = vgg_texture.forward(im1)
    vgg2 = vgg_texture.forward(im2)
    gram1 = [gram_matrix(y) for y in vgg1]
    gram2 = [gram_matrix(y) for y in vgg2]

    for m in range(0, len(vgg1)):
        text_loss += [criterion(gram1[m], gram2[m])]

    loss = torch.log(sum(text_loss))
    return loss.item()

def load_img(filepath):
    from PIL import Image
    img = Image.open(filepath).convert('RGB')
    return torch.stack([ToTensor()(img)])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('im1')
    parser.add_argument('im2')
    parser.add_argument('--texture_layers', nargs='+', default=['8','17','26','35'], help='vgg layers for texture. Default:[]')
    parser.add_argument('--cuda', type=int, default=0, help='Try to use cuda? Default=1')
    opt = parser.parse_args()

    cuda = False
    if opt.cuda:
        if torch.cuda.is_available():
            cuda = True
            torch.cuda.manual_seed(opt.seed)
        else:
            cuda = False
            print('===> Warning: failed to load CUDA, running on CPU!')

    im1 = load_img(opt.im1)
    im2 = load_img(opt.im2)
    print(distance(im1, im2))

