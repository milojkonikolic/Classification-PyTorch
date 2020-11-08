import numpy as np
import torch.nn as nn


def conv_module(input_shape, out_shape, in_channels, out_channels, kernel_size, stride, dilation=1, bias=True):
    """
    Args:
        input_shape: list/tuple, Shape of the input tensor
        out_shape: list/tuple, Shape of the output tensor
        in_channels: int, Number of the input channels
        out_channels: int, Number of the output channels
        kernel_size: int/tuple Size of the kernel
        stride: int/tuple Stride
        dilation: int/tuple, Spacing between kernel elements
        bias: bool, Add bias to the output
    Return:
        Convolution layer
    """

    if isinstance(kernel_size, int):
        kernel_size = np.array([kernel_size, kernel_size])
    if isinstance(stride, int):
        stride = np.array([stride, stride])

    padding = tuple(((np.array(out_shape) * stride - np.array(input_shape) +
                      np.array(dilation) * (kernel_size - 1) + 1) / 2).astype(int))

    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=tuple(kernel_size),
                     stride=tuple(stride),
                     padding=padding,
                     bias=bias)
