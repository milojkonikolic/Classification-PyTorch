import json
import cv2 as cv
import numpy as np
import random
import imutils
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class RandomCrop(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            h, w, _ = img.shape
            ymin = random.randint(0, int(0.3 * h))
            ymax = random.randint(int(0.7 * h), h)
            xmin = random.randint(0, int(0.3 * w))
            xmax = random.randint(int(0.7 * w), w)

            img = img[ymin:ymax, xmin:xmax]
            return img
        else:
            return img


class RandomRotate(object):

    def __init__(self, p=0.5, angle=10):
        self.p = p
        self.angle = angle

    def __call__(self, img):
        if random.random() < self.p:
            img = imutils.rotate(img, self.angle)
            return img
        else:
            return img


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = np.fliplr(img)
            return img
        else:
            return img


class RandomBrightness(object):

    def __init__(self, p=0.5, low_value=0.5, high_value=2.0):
        self.p = p
        self.low_value = low_value
        self.high_value = high_value

    def __call__(self, img):
        value = random.uniform(self.low_value, self.high_value)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 1] = hsv[:, :, 1] * value
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        hsv[:, :, 2] = hsv[:, :, 2] * value
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        hsv = np.array(hsv, dtype=np.uint8)
        img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        return img


class RandomZoom(object):
    pass


class DatasetBuilder(Dataset):

    def __init__(self, data_path, classes_path, img_size, augment, logger):
        self.data_path = data_path
        self.labels = self.load_labels(logger)
        self.img_size = tuple(img_size)
        self.classes = self.get_classes(classes_path)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        label = self.labels[item]
        img_path = label["img_path"]
        class_id = label["label"]
        img = self.read_img(img_path)
        if self.augment:
            img = self.augment_img(img, self.augment)
        img = self.preprocess_img(img, img_path)

        return (img, class_id)

    def get_classes(self, classes_path):
        with open(classes_path, 'r') as f:
            classes = f.read().split('\n')
        return classes

    def load_labels(self, logger):
        with open(self.data_path, 'r') as f:
            labels = json.load(f)
        logger.info(f"Read {len(labels)} images")
        return labels

    def read_img(self, img_path):
        try:
            img = cv.imread(img_path)
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        except:
            img = 0
            print(f"Cannot rad image : {img_path}")
        return img

    @staticmethod
    def augment_img(img, augment):
        try:
            img = RandomCrop(p=augment["RandomCrop"]["p"])(img)
            img = RandomHorizontalFlip(p=augment["RandomHorizontalFlip"]["p"])(img)
            img = RandomRotate(p=augment["RandomRotate"]["p"],
                               angle=augment["RandomRotate"]["angle"])(img)
            img = RandomBrightness(p=augment["RandomBrightness"]["p"],
                                   low_value=augment["RandomBrightness"]["low_value"],
                                   high_value=augment["RandomBrightness"]["high_value"])(img)
        except:
            img = 0

        return img

    def preprocess_img(self, img, img_path):
        try:
            img = cv.resize(img, self.img_size)
            img = np.transpose(img, (2, 0, 1))
            img = img / 255.
        except:
            print(img_path)
        return torch.FloatTensor(img)


def create_dataloaders(train_data_path, val_data_path, classes_path, img_size, batch_size, augment, logger):
    """
    Args:
        train_data_path: str, Path to json file with labels - train dataset
        val_data_path: str, Path to json file with labels - val dataset
        classes_path: str, Path to txt file with class names
        img_size: list, Input shape of the network
        batch_size: int, Batch size
        augment: dict, Augmentation
        logger: logger
    Returns:
        train_loader: Train DataLoader
        val_loader: Validation DataLoader
        classes: List with class names
    """

    logger.info("Reading train data...")
    train_data = DatasetBuilder(train_data_path, classes_path, img_size, augment, logger)
    logger.info("Reading val data...")
    val_data = DatasetBuilder(val_data_path, classes_path, img_size, {}, logger)

    train_loader = DataLoader(dataset=train_data, pin_memory=True,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_data, pin_memory=True,
                            batch_size=batch_size)

    classes = train_data.classes

    return train_loader, val_loader, classes