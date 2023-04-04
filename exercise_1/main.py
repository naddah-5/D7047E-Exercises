from dataset.dataset import Dataset
from model.model import Model
from training.training import Training


def main():
    epochs = 20
    batch_size = 10
    learning_rate = 0.0001
    best_net = ''

    dataset = Dataset(batch_size)
    train_loader, val_loader, test_loader, labels = dataset.load_dataset()
    model = Model()
    training = Training(model, train_loader, val_loader, test_loader, epochs, learning_rate)
    training.train_model()
    best_accuracy = training.test_model()
    print("The best network had the accuracy of %f" % (best_accuracy))

if __name__=="__main__":
    main()