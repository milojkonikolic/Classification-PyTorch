import torch.nn as nn
import torch.nn.functional as F
import torch


def get_padd(kernel_size):

    if isinstance(kernel_size, int):
        return kernel_size // 2
    elif isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
        return (kernel_size[0] / 2, kernel_size[1] / 2)
    else:
        raise ValueError(f"Not supported type for kernel_size: {type(kernel_size)}. Supported types: int, list, tuple")


class ResBlock2(nn.Module):

    def __init__(self, in_channels, out_channels, strides, kernel_size=None):
        super().__init__()
        if not kernel_size:
            kernel_size = [3, 3]

        self.conv1 = nn.Conv2d(in_channels[0], out_channels[0], kernel_size[0],
                               strides[0], padding=get_padd(kernel_size[0]))
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(in_channels[1], out_channels[1], kernel_size[1],
                               strides[1], padding=get_padd(kernel_size[1]))
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


class ResBlock3(nn.Module):

    def __init__(self, in_channels, out_channels, strides, kernel_size=None):
        super().__init__()
        if not kernel_size:
            kernel_size = [1, 3, 1]
        self.conv1 = nn.Conv2d(in_channels[0], out_channels[0], kernel_size[0],
                               strides[0], padding=get_padd(kernel_size[0]))
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(in_channels[1], out_channels[1], kernel_size[1],
                               strides[1], padding=get_padd(kernel_size[1]))
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.conv3 = nn.Conv2d(in_channels[2], out_channels[2], kernel_size[2],
                               strides[2], padding=get_padd(kernel_size[2]))
        self.bn3 = nn.BatchNorm2d(out_channels[2])

        if in_channels[0] != out_channels[2]:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels[0], out_channels[2], 1, strides[0], bias=False),
                                        nn.BatchNorm2d(out_channels[2]))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):

    def __init__(self):
        super().__init__()

    def get_block(self, in_channel, out_channels, stride, num_blocks):
        num_conv = len(out_channels)
        in_channels = [[in_channel, *out_channels[:-1]]] + [[out_channels[-1], *out_channels[:-1]]] * (num_blocks - 1)
        out_channels = [out_channels] * num_blocks
        strides = [[stride, *([1] * (num_conv - 1))]] + [[1] * num_conv] * (num_blocks - 1)
        layers = []
        print(in_channels)
        print(out_channels)
        print(strides)
        print("------------------------")
        for in_channel, out_channel, stride in zip(in_channels, out_channels, strides):
            if len(stride) == 2:
                layer = ResBlock2(in_channels=in_channel, out_channels=out_channel, strides=stride)
            else:
                layer = ResBlock3(in_channels=in_channel, out_channels=out_channel, strides=stride)
            layers.append(layer)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        #x = F.max_pool2d(x, 3, 2)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = F.avg_pool2d(x, 2)
        x = x.view(x.size()[0], -1)
        x = F.log_softmax(self.fc1(x), dim=1)
        return x


class CustomResNet(nn.Module):

    def __init__(self, num_classes, channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 16, 3, 1, padding=get_padd(3))
        self.bn1 = nn.BatchNorm2d(16)
        self.block2 = self.get_block(in_channels=[[16, 16], [16, 16], [16, 16]],
                                     out_channels=[[16, 16], [16, 16], [16, 16]],
                                     strides=[[1, 1, 1], [1, 1, 1]])
        self.block3 = self.get_block(in_channels=[[16, 32], [32, 32], [32, 32]],
                                     out_channels=[[32, 32], [32, 32], [32, 32]],
                                     strides=[[2, 1, 1], [1, 1, 1]])
        self.block4 = self.get_block(in_channels=[[32, 64], [64, 64], [64, 64]],
                                     out_channels=[[64, 64], [64, 64], [64, 64]],
                                     strides=[[2, 1, 1], [1, 1, 1]])
        self.fc1 = nn.Linear(256, num_classes)

    def get_block(self, in_channels, out_channels, strides):
        layers = []
        for in_channel, out_channel, stride in zip(in_channels, out_channels, strides):
            layer = ResBlock2(in_channels=in_channel, out_channels=out_channel, strides=stride)
            layers.append(layer)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size()[0], -1)
        x = F.softmax(self.fc1(x), dim=1)
        return x


class ResNet18(nn.Module):

    def __init__(self, num_classes, input_shape, channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 64, 7, 2, padding=get_padd(7))
        self.bn1 = nn.BatchNorm2d(64)
        self.block2 = self.get_block(in_ch=[64, 64], strides=[1, 1], num_blocks=2)
        self.block3 = self.get_block(in_ch=[64, 128], strides=[2, 1], num_blocks=2)
        self.block4 = self.get_block(in_ch=[128, 256], strides=[2, 1], num_blocks=2)
        self.block5 = self.get_block(in_ch=[256, 512], strides=[2, 1], num_blocks=2)
        self.fc1 = nn.Linear(input_shape[0] // (2 ** 5) * input_shape[1] // (2 ** 5) * 512, num_classes)

    # TO DO: Implement ResNet class with method get_block
    def get_block(self, in_ch, strides, num_blocks):
        in_channels = [in_ch] + [[in_ch[1]] * 2] * (num_blocks - 1)
        out_channels = [[in_ch[1]] * 2] * num_blocks
        strides = [strides] + [[strides[1]] * 2] * (num_blocks - 1)
        layers = []
        for in_channel, out_channel, stride in zip(in_channels, out_channels, strides):
            layer = ResBlock2(in_channels=in_channel, out_channels=out_channel, strides=stride)
            layers.append(layer)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        #x = F.max_pool2d(x, 3, 2)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = F.avg_pool2d(x, 2)
        x = x.view(x.size()[0], -1)
        x = F.log_softmax(self.fc1(x), dim=1)
        return x


class ResNet34(nn.Module):
    def __init__(self, num_classes, input_shape, channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 64, 7, 2, padding=get_padd(7))
        self.bn1 = nn.BatchNorm2d(64)
        self.block2 = self.get_block(in_ch=[64, 64], strides=[1, 1], num_blocks=3)
        self.block3 = self.get_block(in_ch=[64, 128], strides=[2, 1], num_blocks=4)
        self.block4 = self.get_block(in_ch=[128, 256], strides=[2, 1], num_blocks=6)
        self.block5 = self.get_block(in_ch=[256, 512], strides=[2, 1], num_blocks=3)
        self.fc1 = nn.Linear(input_shape[0] // (2 ** 5) * input_shape[1] // (2 ** 5) * 512, num_classes)

    def get_block(self, in_ch, strides, num_blocks):
        in_channels = [in_ch] + [[in_ch[1]] * 2] * (num_blocks - 1)
        out_channels = [[in_ch[1]] * 2] * num_blocks
        strides = [strides] + [[strides[1]] * 2] * (num_blocks - 1)
        layers = []
        for in_channel, out_channel, stride in zip(in_channels, out_channels, strides):
            layer = ResBlock2(in_channels=in_channel, out_channels=out_channel, strides=stride)
            layers.append(layer)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        #x = F.max_pool2d(x, 3, 2)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = F.avg_pool2d(x, 2)
        x = x.view(x.size()[0], -1)
        x = F.log_softmax(self.fc1(x), dim=1)
        return x


class ResNet50(ResNet):
    def __init__(self, num_classes, input_shape, channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 64, 7, 2, padding=get_padd(7))
        self.bn1 = nn.BatchNorm2d(64)
        self.block2 = self.get_block(in_channel=64, out_channels=[64, 64, 256], stride=1, num_blocks=3)
        self.block3 = self.get_block(in_channel=256, out_channels=[128, 128, 512], stride=2, num_blocks=4)
        self.block4 = self.get_block(in_channel=512, out_channels=[256, 256, 1024], stride=2, num_blocks=6)
        self.block5 = self.get_block(in_channel=1024, out_channels=[512, 512, 2048], stride=2, num_blocks=3)
        self.fc1 = nn.Linear(input_shape[0] // (2 ** 5) * input_shape[1] // (2 ** 5) * 2048, num_classes)


class ResNet101(ResNet):
    def __init__(self, num_classes, input_shape, channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 64, 7, 2, padding=get_padd(7))
        self.bn1 = nn.BatchNorm2d(64)
        self.block2 = self.get_block(in_channel=64, out_channels=[64, 64, 256], stride=1, num_blocks=3)
        self.block3 = self.get_block(in_channel=256, out_channels=[128, 128, 512], stride=2, num_blocks=4)
        self.block4 = self.get_block(in_channel=512, out_channels=[256, 256, 1024], stride=2, num_blocks=23)
        self.block5 = self.get_block(in_channel=1024, out_channels=[512, 512, 2048], stride=2, num_blocks=3)
        self.fc1 = nn.Linear(input_shape[0] // (2 ** 5) * input_shape[1] // (2 ** 5) * 2048, num_classes)


class ResNet152(ResNet):
    def __init__(self, num_classes, input_shape, channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 64, 7, 2, padding=get_padd(7))
        self.bn1 = nn.BatchNorm2d(64)
        self.block2 = self.get_block(in_channel=64, out_channels=[64, 64, 256], stride=1, num_blocks=3)
        self.block3 = self.get_block(in_channel=256, out_channels=[128, 128, 512], stride=2, num_blocks=8)
        self.block4 = self.get_block(in_channel=512, out_channels=[256, 256, 1024], stride=2, num_blocks=36)
        self.block5 = self.get_block(in_channel=1024, out_channels=[512, 512, 2048], stride=2, num_blocks=3)
        self.fc1 = nn.Linear(input_shape[0] // (2 ** 5) * input_shape[1] // (2 ** 5) * 2048, num_classes)


if __name__ == "__main__":

    # Test
    net = ResNet18(num_classes=2)(torch.randn(16, 3, 224, 224))
    print(net.shape)
