import torch
import torch.nn as nn
import torchvision.models as VisionModel

class Alex():

    def __init__(self, class_count: int = 10, softmax: bool = False, feature_extract: bool = False):
        alex = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True) 
        in_count = 4096
        
        if feature_extract:
            for parameter in alex.parameters():
                parameter.requires_grad = False
            
        alex.classifier[-1] = nn.Linear(in_count, class_count)

        if softmax:
            self.network = nn.Sequential(
                alex,
                nn.Softmax()
            )
        else:
            self.network = alex

if __name__=="__main__":
    testModel = Alex()