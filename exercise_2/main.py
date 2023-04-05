from dataset.dataset import Dataset
from model.model import Alex
from training.training import Training
import torch

def main():
    epochs = 3
    batch_size = 10
    learning_rate = 0.0001
    best_net = ''

    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # set device to gpu if available

    dataset = Dataset(batch_size)
    #dataset.to(device)
    train_loader, val_loader, test_loader, _ = dataset.load_dataset()
    #train_loader, val_loader, test_loader, _ = train_loader.to(device), val_loader.to(device), test_loader.to(device)
    model = Alex(class_count=10,feature_extract=False,device=my_device)
    #model.to(device)

    training = Training(model.network, train_loader, val_loader, test_loader, epochs, learning_rate,device=my_device)
    training.train_model()
    best_accuracy = training.test_model()
    print("The best network had the accuracy of %f" % (best_accuracy))

if __name__=="__main__":
    main()