import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # TODO: define a small MLP (e.g., Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.01),
            nn.Dropout(p = 0.25),
            nn.Linear(512, 32),
            nn.LeakyReLU(0.01),
            nn.Dropout(p = 0.25),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.01),
            nn.Dropout(p = 0.25),
            nn.Linear(16, 1)
        )
        # TODO: keep output size = 1 for binary classification
        # TODO: try different hidden sizes and dropout rates

    def forward(self, x):
        # TODO: flatten input to (batch, input_dim) and pass through the MLP
        dims = x.size()
        x = torch.flatten(x, 1)
        ret = self.mlp(x)
        return ret

class CNNClassifier(nn.Module):
    def __init__(self, height, width, in_channels=1):
        super().__init__()
        # TODO: define a few conv blocks (Conv2d -> ReLU -> MaxPool)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 4, kernel_size = 1, padding = 0),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(4, 16, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(16, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size = 2)
            # nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            # nn.BatchNorm2d(128),
            # nn.LeakyReLU(0.01),
            # nn.Dropout2d(0.25),
            # nn.MaxPool2d(kernel_size = 2),
            # nn.Conv2d(128, 32, kernel_size = 3, padding = 1),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(0.01),
            # nn.MaxPool2d(kernel_size = 2),
        )
        # TODO: compute the flattened size and add a small classifier head (Linear layers)
        # h = (((height) //2)//2)//2
        # w = (((width) //2)//2)//2
        h = (height) //2//2//2
        w = (width) //2//2//2
        flat = h*w*32
        self.classifier_head = nn.Sequential(
            nn.Linear(flat, flat//128),
            # nn.LeakyReLU(0.01),
            # nn.Dropout(p = 0.25),
            # nn.Linear(flat//8, flat//128),
            nn.LeakyReLU(0.01),
            nn.Dropout(p = 0.25),
            nn.Linear(flat//128, 16),
            nn.LeakyReLU(0.01),
            nn.Dropout(p = 0.25),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # TODO: run conv blocks, flatten, then run the classifier head
        q = self.cnn(x)
        q = torch.flatten(q, 1)
        q = self.classifier_head(q)
        return q
