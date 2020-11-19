from train_detail import train_detail
from torchvision import models
from model.resnest import *


def select_network(network, num_classes=5):

    if network == 'resnest50':
        model = resnest50(pretrained=False, num_classes=num_classes)
        return model

