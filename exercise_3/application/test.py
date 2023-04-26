import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import datetime

def test_model(network, test_loader, epochs: int, learning_rate, best_net = None, device: str='cpu'):
    correct = 0
    total = 0
    accuracy = 0
    for batch_nr, (data, labels) in enumerate(test_loader, 0):
        data, labels=data.to(device), labels.to(device)
        predictions = network.forward(data)

        _, predicted = torch.max(predictions.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy
