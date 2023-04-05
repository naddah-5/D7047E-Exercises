from dataset.dataset import Dataset
from model.model import Alex
from training.training import Training


def main():
    epochs = 3
    batch_size = 10
    learning_rate = 0.0001
    best_net = ''

    dataset = Dataset(batch_size)
    train_loader, val_loader, test_loader, _ = dataset.load_dataset()
    model = Alex(class_count=10)
    training = Training(model.network, train_loader, val_loader, test_loader, epochs, learning_rate)
    training.train_model()
    best_accuracy = training.test_model()
    print("The best network had the accuracy of %f" % (best_accuracy))

if __name__=="__main__":
    main()