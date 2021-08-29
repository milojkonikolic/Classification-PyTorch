import os
import glob
from shutil import copyfile
import torch

from models.CustomNet import CustomNet
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.mobilenets import MobileNetV1, MobileNetV2


def get_model(arch, num_classes, input_shape, channels=3):
    """
    Args:
        arch: string, Network architecture
        num_classes: int, Number of classes
        input_shape: list/tuple, Input shape of the network
        channels: int, Number of input channels
    Returns:
        model, nn.Module, generated model
    """
    if arch == "CustomNet":
        model = CustomNet(num_classes, channels)
    elif arch.lower() == "resnet18":
        model = ResNet18(num_classes, input_shape, channels)
    elif arch.lower() == "resnet34":
        model = ResNet34(num_classes, input_shape, channels)
    elif arch.lower() == "resnet50":
        model = ResNet50(num_classes, input_shape, channels)
    elif arch.lower() == "resnet101":
        model = ResNet101(num_classes, input_shape, channels)
    elif arch.lower() == "resnet152":
        model = ResNet152(num_classes, input_shape, channels)
    elif arch.lower() == "mobilenet_v1":
        model = MobileNetV1(num_classes, channels)
    elif arch.lower() == "mobilenet_v2":
        model = MobileNetV2(num_classes, channels)
    else:
        raise NotImplementedError(f"{arch} not implemented. "
                                  f"For supported architectures see documentation")
    return model


def save_model(model, epoch, batches, ckpt_dir, accuracy, best_accuracy, logger):
    """ Save model
    Args:
        model: Model for saving
        epoch: Number of epoch
        batches: Number of batches
        ckpt_dir: Store directory
        accuracy: Current accuracy
        best_accuracy: Best accuracy
        logger: logger
    """
    ckpt_dir = os.path.join(ckpt_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, "model_epoch_" + str(epoch) + "_batch_" + str(batches) + ".pt")
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Model saved: {ckpt_path}")

    if accuracy == best_accuracy:
        best_ckpt_path = ckpt_path.replace("model_epoch", "best_model_epoch")
        prev_best_model = glob.glob(os.path.join(ckpt_dir, "best_model*"))
        if len(prev_best_model) > 0:
            prev_best_model = prev_best_model[0]
            os.remove(prev_best_model)
        copyfile(ckpt_path, best_ckpt_path)


def load_model(arch, num_classes, input_shape, device, model_path='', channels=3, logger=None):
    """
    Args:
        arch: str, Network architectire (See documentation for supported architectures)
        num_classes: int, Number of classes
        device: device_id
        model_path: str, Path to model
        input_shape: list/tuple, Input shape of the network
        channels: int, Number of input channels
    """

    model = get_model(arch, num_classes, input_shape, channels)
    if model_path:
        logger.info(f"Loading weights from {model_path}...")
        model.load_state_dict(torch.load(model_path))
        model.eval()

    return model.cuda(device)
