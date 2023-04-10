from model.cnn import CNN
import torch
import torch.nn as nn

class FeatureExtractCNN():
    def __init__(self, network: str = "", device: str = "cpu", in_count: int = 21, class_count: int = 10) -> None:
        network = CNN(device=device)
        network = torch.load(network)
        for parameter in network.parameters():
            parameter.requires_grad = False
        network.classifier[-1].requires_grad = True
        self.network = network