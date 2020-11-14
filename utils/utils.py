import os
from shutil import rmtree
import logging
import numpy as np
import cv2 as cv
import torch
from tensorboardX import SummaryWriter

from models.CustomNet import CustomNet
from models.resnet import CustomResNet, ResNet18, ResNet34, ResNet50, ResNet101

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


def get_model(arch, num_classes, input_shape, channels=3):
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
        model = ResNet18(num_classes, input_shape, channels)
    elif arch.lower() == "customresnet":
        model = CustomResNet(num_classes, channels)
    elif arch.lower() == "resnet34":
        model = ResNet34(num_classes, input_shape, channels)
    elif arch.lower() == "resnet50":
        model = ResNet50(num_classes, input_shape, channels)
    elif arch.lower() == "resnet101":
        model = ResNet101(num_classes, input_shape, channels)
    # elif arch.lower() == "resnet152":
    #     model = ResNet152(num_classes, input_shape, channels)
    else:
        raise NotImplementedError(f"{arch} not implemented."
                                  f"For supported architectures see documentation")

    return model


def save_model(model, epoch, ckpt_dir, logger):
    """ Save model
    Args:
        model: Model for saving
        epoch: Number of epoch
        ckpt_dir: Store directory
        logger:
    """
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, "model_epoch" + str(epoch) + ".pt")
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Model saved.")


def load_model(model_path, arch, num_classes, input_shape, channels=3):

    model = get_model(arch, num_classes, input_shape, channels)

    model.load_state_dict(torch.load(model_path))
    model.eval()

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


def preprocess_img(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (240, 160))
    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis, ...]
    img = img / 255.
    img = torch.FloatTensor(img)
    return img
