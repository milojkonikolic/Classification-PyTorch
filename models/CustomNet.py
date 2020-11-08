import torch.nn as nn
import torch.nn.functional as F

from models.common import conv_module


class CustomNet(nn.Module):
    """
    Custom defined net
    """
    # def __init__(self, num_classes, channels=3):
    #     super().__init__()
    #     self.conv1 = nn.Conv2d(channels, 6, 3, 1)
    #     self.conv2 = nn.Conv2d(6, 14, 3, 1)
    #     self.conv3 = nn.Conv2d(14, 20, 3, 1)
    #     self.conv4 = nn.Conv2d(20, 26, 3, 1)
    #     self.fc1 = nn.Linear(12*12*26, 96)
    #     self.fc2 = nn.Linear(96, num_classes)
    #
    # def forward(self, x):
    #     x = F.relu(self.conv1(x))
    #     x = F.max_pool2d(x, 2, 2)
    #     x = F.relu(self.conv2(x))
    #     x = F.max_pool2d(x, 2, 2)
    #     x = F.relu(self.conv3(x))
    #     x = F.max_pool2d(x, 2, 2)
    #     x = F.relu(self.conv4(x))
    #     x = F.max_pool2d(x, 2, 2)
    #     x = x.view(-1, 12*12*26)
    #     x = F.relu(self.fc1(x))
    #     x = F.log_softmax(self.fc2(x), dim=1)
    #
    #     return x

    def __init__(self, num_classes, channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        #self.conv1 = conv_module((224, 224), (224, 224), 3, 6, 3, 1)
        #self.conv2 = conv_module((112, 112), (112, 112), 6, 16, 3, 1)
        self.fc1 = nn.Linear(54 * 54 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 54 * 54 * 16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)
