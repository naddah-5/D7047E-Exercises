import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import datetime

class Test():
    
    def __init__(self, network, test_loader, epochs: int, learning_rate, best_net=None,device: str='cpu'):
        self.network = network
        self.test_loader = test_loader
        self.epochs = epochs
        self.lossfunction = nn.CrossEntropyLoss()
        self.writer = SummaryWriter()
        self.device=device

    def test_model(self):
        correct = 0
        total = 0
        accuracy = 0
        for batch_nr, (data, labels) in enumerate(self.test_loader, 0):
            data, labels=data.to(self.device), labels.to(self.device)
            predictions = self.network.forward(data)

            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = self.lossfunction(predictions, labels)

        accuracy = correct / total
        return accuracy
