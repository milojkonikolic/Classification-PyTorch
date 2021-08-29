import torch.nn as nn
import torch.nn.functional as F


def get_padd(kernel_size):

    if isinstance(kernel_size, int):
        return kernel_size // 2
    elif isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
        return (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        raise ValueError(f"Not supported type for kernel_size: {type(kernel_size)}. Supported types: int, list, tuple")


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=get_padd(kernel_size))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class MobileNetV1(nn.Module):

    def __init__(self, num_classes, channels):
        super().__init__()
        # architecture = [[in_channels, out_channels, kernel_size, stride], ]
        architecture = [(channels, 32, 3, 2), (32, 32, 3, 1), (32, 64, 1, 1), (64, 64, 3, 2), (64, 128, 1, 1),
                        (128, 128, 3, 1), (128, 128, 1, 1), (128, 128, 3, 2), (128, 256, 1, 1), (256, 256, 3, 1),
                        (256, 256, 1, 1), (256, 256, 3, 2), (256, 512, 1, 1), (512, 512, 3, 1), (512, 512, 1, 1),
                        (512, 512, 3, 1), (512, 512, 1, 1), (512, 512, 3, 1), (512, 512, 1, 1), (512, 512, 3, 1),
                        (512, 512, 1, 1), (512, 512, 3, 1), (512, 512, 1, 1), (512, 512, 3, 2), (512, 1024, 1, 1),
                        (1024, 1024, 3, 2), (1024, 1024, 1, 1)]
        conv_layers = []
        for in_channels, out_channels, kernel_size, stride in architecture:
            conv_layers.append(ConvLayer(in_channels, out_channels, kernel_size, stride))
        self.sequential = nn.Sequential(*conv_layers)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.sequential(x)
        x = F.avg_pool2d(x, x.shape[2])
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


# MobileNetV2
class BottleneckResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t):
        super().__init__()
        self.add = in_channels == out_channels
        bottleneck_layers = []
        bottleneck_layers.append(ConvLayer(in_channels, t * in_channels, 1, 1))
        bottleneck_layers.append(ConvLayer(t * in_channels, t * in_channels, 3, stride))
        bottleneck_layers.append(ConvLayer(t * in_channels, out_channels, 1, 1))
        self.bottleneck = nn.Sequential(*bottleneck_layers)

    def forward(self, x):
        out = self.bottleneck(x)
        if self.add:
            return out + x
        else:
            return out


class MobileNetV2(nn.Module):
    def __init__(self, num_classes, channels):
        super().__init__()
        # bottleneck_params = [[in_channels, out_channels, stride, t, n]]
        bottleneck_params = [(32, 16, 1, 1, 1), (16, 24, 2, 6, 2), (24, 32, 2, 6, 3),
                             (32, 64, 2, 6, 4), (64, 96, 1, 6, 3), (96, 160, 2, 6, 3),
                             (160, 320, 1, 6, 1)]
        conv_layers = []
        conv_layers.append(ConvLayer(in_channels=channels, out_channels=32, kernel_size=3, stride=2))
        for in_channels, out_channels, stride, t, n in bottleneck_params:
            for num in range(n):
                conv_layers.append(BottleneckResBlock(in_channels, out_channels, stride, t))
                stride = 1
                in_channels = out_channels
        conv_layers.append(ConvLayer(in_channels=320, out_channels=1280, kernel_size=1, stride=1))
        self.sequential = nn.Sequential(*conv_layers)
        self.last_conv = ConvLayer(in_channels=1280, out_channels=num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.sequential(x)
        x = F.avg_pool2d(x, x.shape[2])
        x = self.last_conv(x)
        x = x.reshape(x.shape[0], -1)
        return x
