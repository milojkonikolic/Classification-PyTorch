import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import conv_module


def inception_module():
    pass


class GoogleNet(nn.Module):

    def __init__(self, num_classes, input_shape, channels=3):
        super().__init__()
        self.conv1 = conv_module(input_shape, (112, 112), channels, 64, 7, 2)
        self.maxpool1 = nn.MaxPool2d(3, 2)
        self.conv2 = conv_module((56, 56), (56, 56), 64, 64, 1, 1)
        self.conv3 = conv_module((56, 56), (56, 56), 64, 192, 3, 1)
        self.maxpool2 = nn.MaxPool2d(3, 2)

        self.inception1 = inception_module()
