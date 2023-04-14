import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import datetime
from application.validate import validate_model


class Training():

    def __init__(self, network, train_loader, val_loader, epochs: int, learning_rate, best_net=None, device: str='cpu'):
        self.network = network
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        self.writer = SummaryWriter()
        self.best_net = best_net
        self.device = device

    def train_model(self):
        best_loss = 100
        #total_iterations = self.epochs * len(self.train_loader)
        iteration = 0
        
        for epoch in range(self.epochs):
            correct = 0
            total = 0
            accuracy = 0
            #start_time = time.time()
            for batch_nr, (data, labels) in enumerate(self.train_loader):
                iteration += 1
                data, labels = data.to(self.device), labels.to(self.device)
                predictions,flattened_layer = self.network.forward(data)

                current_predictions = flattened_layer.cpu().detach().numpy()

                #features = np.concatenate((labels, current_predictions))

                #features = [(label, current_predictions[i]) for i, label in enumerate(labels_list)]

                _, predicted = torch.max(predictions.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = self.loss_function(predictions, labels)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                
                # elapsed_time = time.time() - start_time
                # time_per_iteration = elapsed_time / iteration
                # remaining_iterations = total_iterations - iteration
                # remaining_time = time_per_iteration * remaining_iterations
                # time_in_h=str(datetime.timedelta(seconds=remaining_time))

                print(
                    f'\rEpoch {epoch + 1}/{self.epochs} [{batch_nr + 1}/{len(self.train_loader)}] - Loss {loss:.10f}- Remaining time:' ,
                    end=''
                )

            accuracy = correct / total
            self.writer.add_scalar('Loss/train', loss, (epoch + 1))
            self.writer.add_scalar('Accuracy/train', accuracy, (epoch + 1))
            
            loss, accuracy = validate_model(val_loader=self.val_loader, loss_function=self.loss_function, network=self.network, device=self.device)
            if loss < best_loss:
                best_loss = loss
                torch.save(self.network.state_dict(), "best_network.pt")
            self.writer.add_scalar('Loss/validation', loss, (epoch + 1))
            self.writer.add_scalar('Accuracy/validation', accuracy, (epoch + 1))

        return flattened_layer, current_predictions
