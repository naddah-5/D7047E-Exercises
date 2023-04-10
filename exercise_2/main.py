from dataset.dataset import Dataset
from model.transfer_learning import TransferModel
from model.alex import Alex
from model.cnn import CNN
from training.training import Training
import torch

def main():
    epochs = 1
    batch_size = 10
    learning_rate = 0.0001
    best_net: str = ''

    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set device to gpu if available
    print("using device ", my_device)


    datasetMNIST = Dataset(batch_size, MNIST=True)
    datasetSVHN = Dataset(batch_size, SVHN=True)
    
    train_loader_MNIST, val_loader_MNIST, test_loader_MNIST, _ = datasetMNIST.load_dataset()
    train_loader_SVHN, val_loader_SVHN, test_loader_SVHN, _ = datasetSVHN.load_dataset()

    #model = CNN(class_count=10, device=my_device)
    model = TransferModel()
    model.load_CNN(network_path="best_network.pt")
    model.network.to(my_device)

    training = Training(model.network, train_loader_MNIST, val_loader_MNIST, test_loader_MNIST, test_loader_SVHN, epochs, learning_rate, device=my_device)
    training.train_model()
    

    best_accuracy_MNIST = training.test_model_MNIST()
    best_accuracy_SVHN = training.test_model_SVHN()

    print("\n MNIST Test accuracy of %f" % (best_accuracy_MNIST))
    print("SVHN Test accuracy of %f" % (best_accuracy_SVHN))

if __name__=="__main__":
    main()
