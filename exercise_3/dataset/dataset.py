import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import ssl 

class Dataset():

    def __init__(self, batch_size):
        self.batch_size = batch_size
        ssl._create_default_https_context = ssl._create_unverified_context # fixed certification error

    def load_dataset(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_val_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        proportions = [.40, .60]
        lengths = [int(p * len(test_val_set)) for p in proportions]
        lengths[-1] = len(test_val_set) - sum(lengths[:-1])

        valset , testset = torch.utils.data.random_split(test_val_set, lengths)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        valloader = torch.utils.data.DataLoader(valset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        labels = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

        return trainloader, valloader, testloader, labels