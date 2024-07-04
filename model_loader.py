from email.header import Header
import math
import torch
import torch.nn.functional as F
from torch import nn
from resnet import ResNet18, ResNet50
from resnet import BACKBONES 
from resnet import HEADS
from torch.nn import Sequential
from collections import OrderedDict
import torchvision

def load_model(args):
    if args.arch == 'resnet50':
        backbone = BACKBONES[args.arch](last_stride=args.last_stride) # resnet50
        head = HEADS['linear'](args)
        model = Sequential(OrderedDict([("backbone", backbone), ("head", head)]))
        model.backbone.load_param('./resnet50-19c8e357.pth')
    else:
        backbone = torchvision.models.resnet18(pretrained=True) # resnet18
        head = HEADS['linear'](args)
        model = Sequential(OrderedDict([("backbone", backbone), ("head", head)]))
    return model


