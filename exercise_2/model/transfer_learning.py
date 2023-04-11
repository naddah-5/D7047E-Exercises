from model.cnn import CNN
import torch
import torch.nn as nn

class TransferModel():
    def __init__(self) -> None:
        self.network = None
    
    
    def load_CNN(self, network_path: str = "", device: str = "cpu", in_count: int = 21, class_count: int = 10) -> None:
        network = CNN(device=device)
        dict = torch.load(network_path)
        network.load_state_dict(dict)
        #for parameter in network.parameters():
        #    parameter.requires_grad = False
        network.classifier[-1] = nn.Linear(in_count, class_count)
        self.network = network