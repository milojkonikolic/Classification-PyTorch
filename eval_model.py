import os
import json
import argparse
import yaml
import random
import cv2 as cv
import torch

from utils.utils import load_model, preprocess_img, get_device


def evaluate(model, data, device):

    correct = 0
    for num, d in enumerate(data):
        print(f"Progress: {num+1}/{len(data)}", end='\r')
        img_path = d["img_path"]
        label = d["label"]
        img = cv.imread(img_path)
        img = preprocess_img(img)
        img = img.cuda(device)
        pred = model(img)
        predicted = int(torch.max(pred.data, 1)[1])

        if predicted == label:
            correct += 1
        # print(f"label: {label}")
        # print(f"pred: {predicted}\n")
    acc = correct / len(data)
    print(f"Accuracy: {acc}")


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

    device = get_device(args.device)
    model = load_model(args.model, arch=config["Train"]["arch"],
                       num_classes=config["Dataset"]["num_classes"],
                       input_shape=config["Train"]["image_size"],
                       channels=config["Train"]["channels"]
                       ).cuda(device)

    if os.path.isfile(args.dataset):
        with open(args.dataset, 'r') as data_file:
            data = json.load(data_file)
    else:
        raise ValueError(f"Path {args.dataset} not found")

    random.shuffle(data)
    evaluate(model, data, device)
