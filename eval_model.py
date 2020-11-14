import os
import json
import argparse
import yaml
import cv2 as cv

from utils.utils import load_model, preprocess_img


def evaluate(model, data):

    for d in data:
        img_path = d["img_path"]
        label = d["label"]
        img = cv.imread(img_path)
        img = preprocess_img(img)
        pred = model(img)
        print(f"label: {label}")
        print(f"pred: {pred}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='',
                        help="Path to model")
    parser.add_argument("--dataset", type=str, default='',
                        help="Path to dataset")
    parser.add_argument("--config", type=str, default='',
                        help="Path to config file")
    args = parser.parse_args()

    if os.path.isfile(args.config):
        with open(args.config, 'r') as cfg_file:
            config = yaml.load(cfg_file, Loader=yaml.FullLoader)
    else:
        raise ValueError(f"Path {args.config} not found")

    model = load_model(args.model, arch=config["Train"]["arch"],
                       num_classes=config["Dataset"]["num_classes"],
                       input_shape=config["Train"]["image_size"],
                       channels=config["Train"]["channels"])

    if os.path.isfile(args.dataset):
        with open(args.dataset, 'r') as data_file:
            data = json.load(data_file)
    else:
        raise ValueError(f"Path {args.dataset} not found")

    evaluate(model, data[:5])
