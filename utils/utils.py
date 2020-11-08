import os
from shutil import rmtree
import logging

import torch
from tensorboardX import SummaryWriter

from models.CustomNet import CustomNet
from models.resnet import ResNet18

import warnings
warnings.filterwarnings("ignore")


def get_logger():
    logger = logging.getLogger("ClassNets")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s]-[%(filename)s]: %(message)s ")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def get_tb_writer(tb_logdir, ckpt_dir):
    tb_logdir = os.path.join(tb_logdir, ckpt_dir.split("\\")[-1])
    if os.path.isdir(tb_logdir):
        rmtree(tb_logdir)
    os.mkdir(tb_logdir)
    writer = SummaryWriter(log_dir=tb_logdir)
    return writer


def get_device(device):
    """
    Args:
        device: str, GPU device id
    Return: torch device
    """

    if device == "cpu":
        return torch.device("cpu")
    else:
        assert torch.cuda.is_available(), f"CUDA unavailable, invalid device {device} requested"
        c = 1024 ** 2
        x = torch.cuda.get_device_properties(0)
        print("Using GPU")
        print(f"device{device} _CudaDeviceProperties(name='{x.name}'"
              f", total_memory={x.total_memory / c}MB)")
        return torch.device("cuda:0")


def get_model(arch, num_classes, channels=3):
    """
    Args:
        arch: string, Network architecture
        num_classes: int, Number of classes
        channels: int, Number of input channels
    Returns:
        model, nn.Module, generated model
    """
    if arch == "CustomNet":
        model = CustomNet(num_classes, channels)
    elif arch.lower() == "resnet18":
        model = ResNet18(num_classes, channels)
    else:
        raise NotImplementedError(f"{arch} not implemented."
                                  f"For supported architectures see documentation")

    return model


def get_optimizer(opt, model, lr):
    """
    Args
        opt: string, optimizer from config file
        model: nn.Module, generated model
        lr: float, specified learning rate
    Returns:
        optimizer
    """

    if opt.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise NotImplementedError(f"Not supported optimizer name: {opt}."
                                  f"For supported optimizers see documentation")
    return optimizer


def save_model(model, ckpt_path):
    """ Save model
    Args:
        model: Model for saving
        ckpt_path: Path to saved model
    """
    torch.save(model.state_dict, ckpt_path)
