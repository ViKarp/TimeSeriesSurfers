import torch
import torch.optim as optim


class BaseTrainer:
    def __init__(self):
        pass

    def train(self):
        pass


class TorchTrainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, train_loader, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
