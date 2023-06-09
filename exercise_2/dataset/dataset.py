import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import ssl 

class Dataset():

    def __init__(self, batch_size, MNIST: bool = False, SVHN: bool = False, CIFAR10: bool = False, scale: int = 32):
        if not MNIST ^ SVHN ^ CIFAR10:
            CIFAR10 = True
            MNIST = False
            SVHN = False
            
        self.MNIST = MNIST
        self.SVHN = SVHN
        self.CIFAR10 = CIFAR10
        self.scale = scale
        self.batch_size = batch_size
        ssl._create_default_https_context = ssl._create_unverified_context # fixed certification error

    def load_dataset(self):
        transform = transforms.Compose([
            transforms.Resize(self.scale),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if self.MNIST:
            trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            test_val_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        elif self.SVHN:
            trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
            test_val_set = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
        elif self.CIFAR10:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            test_val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        

        proportions = [.40, .60]
        lengths = [int(p * len(test_val_set)) for p in proportions]
        lengths[-1] = len(test_val_set) - sum(lengths[:-1])

        valset , testset = torch.utils.data.random_split(test_val_set, lengths)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        valloader = torch.utils.data.DataLoader(valset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        
       
        labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return trainloader, valloader, testloader, labels
    