from dataset.dataset import Dataset
from model.alex import Alex
from model.cnn import CNN
from training.training import Training
import torch

def main():
    epochs = 30
    batch_size = 10
    learning_rate = 0.0001
    best_net = ''

    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set device to gpu if available

    dataset = Dataset(batch_size, SVHN=True)
    train_loader, val_loader, test_loader, _ = dataset.load_dataset()

    model = CNN(class_count=10, device=my_device)
    model.to(my_device)
    training = Training(model, train_loader, val_loader, test_loader, epochs, learning_rate, device=my_device)
    training.train_model()
    best_accuracy = training.test_model()
    print("The best network had the accuracy of %f" % (best_accuracy))

if __name__=="__main__":
    main()
