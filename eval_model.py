import os
import argparse
import yaml
import torch
from tqdm import tqdm

from utils.dataset import DatasetBuilder
from utils.utils import get_device, get_logger
from models.common import load_model


def evaluate(model, dataset, device):
    """ Get predictions from model and calculate accuracy
    Args:
        model: PyTorch Model
        data: List of images
        device: cuda or cpu
    Return:

    """

    correct = 0
    for num in tqdm(range(len(dataset))):
        img, label = dataset[num]
        img = img.unsqueeze(0).cuda(device)
        pred = model(img)
        predicted = int(torch.max(pred.data, 1)[1])
        if predicted == label:
            correct += 1
    acc = int(correct / len(dataset) * 100)
    print(f"Accuracy: {acc} %")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='',
                        help="Path to model")
    parser.add_argument("--dataset", type=str, default='',
                        help="Path to dataset")
    parser.add_argument("--config", type=str, default='',
                        help="Path to config file")
    parser.add_argument("--device", type=str, default='0',
                        help="Device: 'cpu', '0', '1', ...")
    args = parser.parse_args()

    if os.path.isfile(args.config):
        with open(args.config, 'r') as cfg_file:
            config = yaml.load(cfg_file, Loader=yaml.FullLoader)
    else:
        raise ValueError(f"Path {args.config} not found")

    logger = get_logger()
    device = get_device(args.device)
    model = load_model(arch=config["Train"]["arch"], num_classes=config["Dataset"]["num_classes"],
                       input_shape=config["Train"]["image_size"], device=device, model_path=args.model,
                       channels=config["Train"]["channels"], logger=logger)

    if os.path.isfile(args.dataset):
        dataset = DatasetBuilder(args.dataset, config["Dataset"]["classes_path"], config["Train"]["image_size"], logger)
    else:
        raise ValueError(f"Path {args.dataset} not found")

    evaluate(model, dataset, device)
