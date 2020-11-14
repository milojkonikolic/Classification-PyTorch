import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import cv2 as cv

from utils.utils import get_model, get_optimizer, get_logger, get_tb_writer, get_device, save_model
from utils.dataset import create_dataloaders

logger = get_logger()


def train(config):

    writer = get_tb_writer(config["Logging"]["tb_logdir"], config["Logging"]["ckpt_dir"])

    # Make train and val DataLoaders
    train_loader, val_loader, classes = create_dataloaders(train_data_path=config["Dataset"]["train_data_path"],
                                                           val_data_path=config["Dataset"]["val_data_path"],
                                                           classes_path=config["Dataset"]["classes_path"],
                                                           img_size=config["Train"]["image_size"],
                                                           batch_size=config["Train"]["batch_size"],
                                                           augment=config["Augmentation"],
                                                           logger=logger)
    device = get_device(config["Train"]["device"])
    model = get_model(arch=config["Train"]["arch"],
                      num_classes=config["Dataset"]["num_classes"],
                      input_shape=config["Train"]["image_size"],
                      channels=config["Train"]["channels"]
                      ).cuda(device=device)

    optimizer = get_optimizer(opt=config["Train"]["optimizer"], model=model,
                              lr=config["Train"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    params = [p.numel() for p in model.parameters() if p.requires_grad]
    total_num_params = 0
    for p in params:
        total_num_params += p
    logger.info(f"Number of parameters: {total_num_params}")

    global_step = 0
    logger.info(f"---------------- Training Started ----------------")
    for epoch in range(config["Train"]["epochs"]):
        epoch += 1
        for batch, (X_train, y_train) in enumerate(train_loader):
            batch += 1

            X_train = X_train.cuda()
            y_train = y_train.cuda()
            y_pred = model(X_train)
            train_loss = criterion(y_pred, y_train)
            predicted = torch.max(y_pred.data, 1)[1]

            global_step += batch
            writer.add_scalar("train_loss", train_loss.item(), global_step)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                logger.info(f"epoch: {epoch}/{config['Train']['epochs']}, batch: {batch}/{len(train_loader)}"
                            f", train_loss: {train_loss.item()}")

        val_corr = 0
        with torch.no_grad():
            for batch, (X_val, y_val) in enumerate(val_loader):
                batch += 1
                X_val = X_val.cuda()
                y_val = y_val.cuda()
                y_pred = model(X_val)
                predicted = torch.max(y_pred.data, 1)[1]
                val_corr += (predicted == y_val).sum()

            val_loss = criterion(y_pred, y_val)
            logger.info(f"Val loss: {val_loss.item()}")
            writer.add_scalar("val_loss", val_loss.item(), global_step)
            acc = int(float(val_corr) / (float(config["Train"]["batch_size"])
                      * float(len(val_loader))) * 100.)
            logger.info(f"val_accuracy: {acc}%")
            writer.add_scalar("val_accuracy", acc, epoch)

            save_model(model, epoch, config["Logging"]["ckpt_dir"], logger)

    logger.info(f"---------------- Training Finished ----------------")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='',
                        help="Path to config file")
    args = parser.parse_args()

    # Get parameters from the config file
    with open(args.config, 'r') as cfg_file:
        config = yaml.load(cfg_file, Loader=yaml.FullLoader)

    train(config)
