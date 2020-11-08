import torch.nn as nn
import torch.nn.functional as F
import torch


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, strides):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels[0], out_channels[0], 3, strides[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(in_channels[1], out_channels[1], 3, strides[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels[1])

        if in_channels[0] != out_channels[1]:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels[0], out_channels[1], 1, strides[0], bias=False),
                                        nn.BatchNorm2d(out_channels[1]))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):

    def __init__(self, num_classes, channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 64, 7, 2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.block2 = self.get_block(in_channels=[[64, 64], [64, 64]], out_channels=[[64, 64], [64, 64]],
                                     strides=[[1, 1], [1, 1]])
        self.block3 = self.get_block(in_channels=[[64, 128], [128, 128]], out_channels=[[128, 128], [128, 128]],
                                     strides=[[2, 1], [1, 1]])
        self.block4 = self.get_block(in_channels=[[128, 256], [256, 256]], out_channels=[[256, 256], [256, 256]],
                                     strides=[[2, 1], [1, 1]])
        self.block5 = self.get_block(in_channels=[[256, 512], [512, 512]], out_channels=[[512, 512], [512, 512]],
                                     strides=[[2, 1], [1, 1]])
        self.fc1 = nn.Linear(512, num_classes)

    # TO DO: Implement ResNet class with method get_block
    def get_block(self, in_channels, out_channels, strides):
        layers = []
        for in_channel, out_channel, stide in zip(in_channels, out_channels, strides):
            layer = ResBlock(in_channels=in_channel, out_channels=out_channel, strides=stide)
            layers.append(layer)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 3, 2)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size()[0], -1)
        x = F.log_softmax(self.fc1(x), dim=1)
        return x


if __name__ == "__main__":

    # Test
    net = ResNet18(num_classes=2)(torch.randn(16, 3, 224, 224))
    print(net.shape)
