import torch.nn as nn
import torch.nn.functional as F


class CustomNet(nn.Module):
    """
    Custom defined net
    """

    def __init__(self, num_classes, channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1, padding=1)
        #self.conv1 = conv_module((224, 224), (224, 224), 3, 6, 3, 1)
        #self.conv2 = conv_module((112, 112), (112, 112), 6, 16, 3, 1)
        self.fc1 = nn.Linear(56 * 56 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 56 * 56 * 16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)
