import os
import yaml
import logging
import torch
from tensorboardX import SummaryWriter


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


def get_tb_writer(tb_logdir):
    """
    Args:
        tb_logdir: str, Path to directory fot tensorboard events
    Return:
        writer: TensorBoard writer
    """
    if not os.path.isdir(tb_logdir):
        os.makedirs(tb_logdir)
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


def copy_config(config):
    """ Copy used config for training to checkpoints directory
    Args:
    config: Config file
    """
    ckpt_dir = config["Logging"]["ckpt_dir"]
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    out_config_path = os.path.join(ckpt_dir, "config.yaml")
    with open(out_config_path, 'w') as outfile:
        yaml.dump(config, outfile)
