import os
import argparse
from shutil import copyfile
import yaml
import torch
import torch.nn as nn

from utils.dataset import create_dataloaders
from utils.utils import get_device, get_logger, get_tb_writer, copy_config
from utils.dataset import create_dataloaders
from models.CustomNet import CustomNet
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


class Train():
    def __init__(self, config):
        self.logger = get_logger()
        self.writer = get_tb_writer(config["Logging"]["tb_logdir"], config["Logging"]["ckpt_dir"])
        self.num_classes = config["Dataset"]["num_classes"]
        self.batch_size = config["Train"]["batch_size"]
        self.epochs = config["Train"]["epochs"]
        self.input_shape = config["Train"]["image_size"]
        self.channels = config["Train"]["channels"]
        self.ckpt_dir = config["Train"]["ckpt_dir"]
        self.device = get_device(config["Train"]["device"])
        self.model = self.load_model(config["Train"]["arch"], config["Train"]["pretrained"])
        self.optimizer = self.get_optimizer(config["Train"]["optimizer"], config["Train"]["lr_init"])
        self.train_loader, self.val_loader, self.classes = \
            create_dataloaders(train_data_path=config["Dataset"]["train_data_path"],
                               val_data_path=config["Dataset"]["val_data_path"],
                               classes_path=config["Dataset"]["classes_path"],
                               img_size=self.input_shape,
                               batch_size=self.batch_size,
                               augment=config["Augmentation"],
                               logger=self.logger)
        self.criterion = nn.CrossEntropyLoss()


    def get_optimizer(self, opt, lr=0.001):
        """
        Args
            opt: string, optimizer from config file
            model: nn.Module, generated model
            lr: float, specified learning rate
        Returns:
            optimizer
        """

        if opt.lower() == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif opt.lower() == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError(f"Not supported optimizer name: {opt}."
                                      f"For supported optimizers see documentation")
        return optimizer

    def get_model(self, arch):
        """
        Args:
            arch: string, Network architecture
        Returns:
            model, nn.Module, generated model
        """
        if arch == "CustomNet":
            model = CustomNet(self.num_classes, self.channels)
        elif arch.lower() == "resnet18":
            model = ResNet18(self.num_classes, self.input_shape, self.channels)
        elif arch.lower() == "resnet34":
            model = ResNet34(self.num_classes, self.input_shape, self.channels)
        elif arch.lower() == "resnet50":
            model = ResNet50(self.num_classes, self.input_shape, self.channels)
        elif arch.lower() == "resnet101":
            model = ResNet101(self.num_classes, self.input_shape, self.channels)
        elif arch.lower() == "resnet152":
            model = ResNet152(self.num_classes, self.input_shape, self.channels)
        else:
            raise NotImplementedError(f"{arch} not implemented."
                                      f"For supported architectures see documentation")
        return model

    def load_model(self, arch, pretrained=''):

        model = self.get_model(arch)
        if pretrained:
            model.load_state_dict(torch.load(pretrained))
            model.eval()

        return model.cuda(self.device)

    def save_model(self, epoch, results):
        """ Save model
        Args:
            epoch: Number of epoch
            results: list, List of dicts with results
        """
        best_acc = 0
        best_epoch = 0
        ckpt_dir = os.path.join(self.ckpt_dir, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        ckpt_path = os.path.join(ckpt_dir, "model_epoch" + str(epoch) + ".pt")
        torch.save(self.model.state_dict(), ckpt_path)
        self.logger.info(f"Model saved: {ckpt_path}")

        best_ckpt_path = os.path.join(ckpt_dir, "best_model_epoch" + str(epoch) + ".pt")
        for res in results:
            if res["val_accuracy"] > best_acc:
                best_epoch = res["epoch"]
        best_epoch = os.path.join(ckpt_dir, "model_epoch" + str(best_epoch) + ".pt")
        if os.path.isfile(best_epoch):
            copyfile(best_epoch, best_ckpt_path)
        else:
            self.logger.info(f"The best epoch not found: {best_epoch}")

    def train(self):

        params = [p.numel() for p in self.model.parameters() if p.requires_grad]
        total_num_params = 0
        for p in params:
            total_num_params += p
        self.logger.info(f"Number of parameters: {total_num_params}")

        global_step = 0
        self.logger.info(f"---------------- Training Started ----------------")
        results = []

        for epoch in range(self.epochs):
            epoch += 1
            for batch, (X_train, y_train) in enumerate(self.train_loader):
                batch += 1

                X_train = X_train.cuda(self.device)
                y_train = y_train.cuda(self.device)
                y_pred = self.model(X_train)
                train_loss = self.criterion(y_pred, y_train)

                global_step += batch
                self.writer.add_scalar("train_loss", train_loss.item(), global_step)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                if batch % 100 == 0:
                    self.logger.info(f"epoch: {epoch}/{self.epochs}, "
                                     f"batch: {batch}/{len(self.train_loader)}"
                                     f", train_loss: {train_loss.item()}")

            val_corr = 0
            with torch.no_grad():
                for batch, (X_val, y_val) in enumerate(self.val_loader):
                    batch += 1
                    X_val = X_val.cuda(self.device)
                    y_val = y_val.cuda(self.device)
                    y_pred = self.model(X_val)
                    predicted = torch.max(y_pred.data, 1)[1]
                    val_corr += (predicted == y_val).sum()

                val_loss = self.criterion(y_pred, y_val)
                self.logger.info(f"Val loss: {val_loss.item()}")
                self.writer.add_scalar("val_loss", val_loss.item(), global_step)
                acc = int(float(val_corr) / (float(self.batch_size) * float(len(self.val_loader))) * 100.)
                self.logger.info(f"val_accuracy: {acc}%")
                self.writer.add_scalar("val_accuracy", acc, epoch)
                results.append({"epoch": epoch, "val_accuracy": acc})
                self.save_model(epoch, results)

        self.logger.info(f"---------------- Training Finished ----------------")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='',
                        help="Path to config file")
    args = parser.parse_args()

    # Get parameters from the config file
    with open(args.config, 'r') as cfg_file:
        config = yaml.load(cfg_file, Loader=yaml.FullLoader)

    copy_config(config)

    Trainer = Train(config)
    Trainer.train()