import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    #    self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(nn.LeakyReLU()(self.conv1(x)))
        x = self.pool(nn.LeakyReLU()(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = nn.LeakyReLU()(self.fc1(x))
        x = nn.LeakyReLU()(self.fc2(x))
        x = self.fc3(x)
        #    x = self.soft(x)
        return x

    # def forward(self, x):
    #     x = self.pool(nn.Tanh()(self.conv1(x)))
    #     x = self.pool(nn.Tanh()(self.conv2(x)))
    #     x = torch.flatten(x, 1)  # flatten all dimensions except batch
    #     x = nn.Tanh()(self.fc1(x))
    #     x = nn.Tanh()(self.fc2(x))
    #     x = self.fc3(x)
    #     #    x = self.soft(x)
    #     return x