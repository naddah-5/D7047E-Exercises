from dataset.dataset import Dataset
from model.nnc import CNN
from application.train import Training
import torch


def main():
    epochs = 1
    batch_size = 10
    learning_rate = 0.000001
    best_net: str = ''

    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set device to gpu if available
    print("using device ", my_device)


    datasetMNIST = Dataset(batch_size, MNIST=True)

    train_loader_MNIST, val_loader_MNIST, test_loader_MNIST, _ = datasetMNIST.load_dataset()
    

    model = CNN(class_count=10, device=my_device)
    model.to(my_device)

    training = Training(model, train_loader_MNIST, val_loader_MNIST, epochs, learning_rate, device=my_device)
    flattened_layer=training.train_model()



    

    best_accuracy = training.test_model()

    print("\n CIFAR Test accuracy of %f" % (best_accuracy))


if __name__=="__main__":
    main()
