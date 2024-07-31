import torch.nn as nn
import torch
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from torch import optim


class BaseModel(ABC):
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def evaluate(self, X, y):
        pass


class NNModel(BaseModel):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, X, y, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            inputs = torch.tensor(X, dtype=torch.float32)
            labels = torch.tensor(y, dtype=torch.long)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        print(f"Training finished for {epochs} epochs")

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = (predictions == y).sum() / len(y)
        print(f'Accuracy: {accuracy * 100}%')
        return accuracy


class SklearnModel(BaseModel):
    def __init__(self, model: BaseEstimator):
        self.model = model

    def train(self, X, y):
        self.model.fit(X, y)
        print("Training finished")

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = (predictions == y).sum() / len(y)
        print(f'Accuracy: {accuracy * 100}%')
        return accuracy
