import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

def importData():
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainset, valset = torch.utils.data.random_split(trainset, [40000, 10000])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    return trainloader, valloader, testloader

def visulizeData(data):
    # Define the class labels
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Get some random training images
    dataiter = iter(data)
    images, labels = dataiter.next()

    # Convert images and labels to numpy arrays
    images = images.numpy()
    labels = labels.numpy()

    # Plot the images with their labels
    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(12, 6))

    for i, ax in enumerate(axes.flat):
        ax.imshow(np.transpose(images[i], (1, 2, 0)))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(classes[labels[i]])

    plt.show()

trainl, vall, testl = importData()
visulizeData(trainl)
