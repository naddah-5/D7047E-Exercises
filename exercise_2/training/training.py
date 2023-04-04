import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class Training():

    def __init__(self, network, train_loader, val_loader, test_loader, epochs: int, learning_rate, best_net=None):
        self.network = network
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.lossfunction = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        self.writer = SummaryWriter()
        self.best_net = best_net


    def train_model(self):
        best_loss = 100
        for epoch in range(self.epochs):
            correct = 0
            total = 0
            accuracy = 0
            for batch_nr, (data, labels) in enumerate(self.train_loader):

                predictions = self.network.forward(data)

                _, predicted = torch.max(predictions.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = self.lossfunction(predictions, labels)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                print(
                    f'\rEpoch {epoch + 1} [{batch_nr + 1}/{len(self.train_loader)}] - Loss {loss}',
                    end=''
                )

            accuracy = correct / total
            self.writer.add_scalar('Loss/train', loss, (epoch + 1))
            self.writer.add_scalar('Accuracy/train', accuracy, (epoch + 1))

            best_loss = self.val_model(epoch, best_loss)

        return ()

    def val_model(self, epoch, best_loss):
        correct = 0
        total = 0
        accuracy = 0
        for batch_nr, (data, labels) in enumerate(self.val_loader, 0):
            predictions = self.network.forward(data)

            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = self.lossfunction(predictions, labels)

        accuracy = correct / total
        self.writer.add_scalar('Loss/validation', loss, (epoch + 1))
        self.writer.add_scalar('Accuracy/validation', accuracy, (epoch + 1))

        if loss < best_loss:
            best_loss = loss
            torch.save(self.network, "best_network.pt")

        return best_loss

    def test_model(self):
        correct = 0
        total = 0
        accuracy = 0
        for batch_nr, (data, labels) in enumerate(self.test_loader, 0):
            predictions = self.network.forward(data)

            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = self.lossfunction(predictions, labels)

        accuracy = correct / total

        return accuracy

