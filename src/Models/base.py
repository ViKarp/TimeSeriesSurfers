import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import RegressorMixin
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model on the provided training data.

        :param X_train: Training features.
        :param y_train: Training labels.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Makes predictions using the model on the provided data.

        :param X: Features to make predictions on.
        :return: Predictions.
        """
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        """
        Evaluates the model on the provided test data.

        :param X_test: Test features.
        :param y_test: Test labels.
        :return: Evaluation metric(s).
        """
        pass


class TorchModel(BaseModel):
    def __init__(self, model, loss_fn=nn.MSELoss(), optimizer_cls=optim.Adam, lr=0.001, batch_size=32, epochs=100):
        """
        Initializes the TorchModel with the given parameters.

        :param model: PyTorch model to be trained.
        :param loss_fn: Loss function.
        :param optimizer_cls: Optimizer class.
        :param lr: Learning rate.
        """

        class DataDataset(Dataset):
            def __init__(self, X, y):
                self.X = torch.from_numpy(X).float()
                self.y = torch.from_numpy(y).float()

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = optimizer_cls(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataset = DataDataset

    def train(self, X_train, y_train):
        """
        Trains the model on the provided training data.

        :param X_train: Training features.
        :param y_train: Training labels.
        :param epochs: Number of training epochs.
        :param batch_size: Batch size for training.
        """

        self.model.train()
        dataset = self.dataset(X_train.values, y_train.values)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)
                loss = self.loss_fn(pred, y)

                # Backpropagation.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch % 100 == 0:
                    loss, current = loss.item(), batch
                    # TODO: log metric? MLflow? Custom?
                    # step = batch // 100 * (epoch + 1)
                    print(f"loss: {loss:2f} [{current} / {len(dataloader)}]")

    def predict(self, X):
        """
        Makes predictions using the model on the provided data.

        :param X: Features to make predictions on.
        :return: Predictions.
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            predictions = self.model(X_tensor)
        return predictions.numpy()

    def evaluate(self, X_val, y_val):
        """
        Evaluate the model on a single pass of the dataloader.

        Args:
            :param X_val: Training features.
            :param y_val: Training labels.
        """
        dataset = self.dataset(X_val.values, y_val.values)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        num_batches = len(dataloader)
        self.model.eval()
        eval_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                eval_loss += self.loss_fn(pred, y).item()

        eval_loss /= num_batches

        return eval_loss

    def train_step(self, dataloader, epoch):
        """Train the model on a single pass of the dataloader.

        Args:
            dataloader: an instance of `torch.utils.data.DataLoader`, containing the training data.
            epoch: an integer, the current epoch number.
        """
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch
                # TODO: log metric? MLflow? Custom?
                # step = batch // 100 * (epoch + 1)
                print(f"loss: {loss:2f} [{current} / {len(dataloader)}]")
    def train_evaluate(self, train_dataloader, val_dataloader, epochs=100, batch_size=32):
        # TODO: function with train_step and evaluate. End of training with EarlyStopping
        pass


class SKLearnModel(BaseModel):
    def __init__(self, model=LinearRegression):
        """
        Initializes the SKLearnModel with the given scikit-learn model.

        :param model: Scikit-learn model to be trained.
        """
        self.model = model()

    def train(self, X_train, y_train):
        """
        Trains the model on the provided training data.

        :param X_train: Training features.
        :param y_train: Training labels.
        """
        self.model.fit(X_train, y_train.values)

    def predict(self, X):
        """
        Makes predictions using the model on the provided data.

        :param X: Features to make predictions on.
        :return: Predictions.
        """
        return self.model.predict(X)

    def evaluate(self, X_test, y_test, metric=mean_squared_error):
        """
        Evaluates the model on the provided test data.

        :param X_test: Test features.
        :param y_test: Test labels.
        :param metric: metric to evaluate the model.
        :return: Mean squared error on the test data.
        """
        predictions = self.predict(X_test)
        return metric(y_test.values, predictions)
