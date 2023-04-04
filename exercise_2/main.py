from exercise_2.dataset.dataset import Dataset
from exercise_2.model.model import AlexNet
from exercise_2.training.training import Training


def main():
    epochs = 20
    batch_size = 10
    learning_rate = 0.0001
    best_net = ''

    dataset = Dataset(batch_size)
    train_loader, val_loader, test_loader, labels = dataset.load_dataset()
    model = AlexNet(num_classes=10)
    training = Training(model, train_loader, val_loader, test_loader, epochs, learning_rate)
    training.train_model()
    best_accuracy = training.test_model()
    print("The best network had the accuracy of %f" % (best_accuracy))

if __name__=="__main__":
    main()