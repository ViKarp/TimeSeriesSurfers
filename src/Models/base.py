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
    def __init__(self, logger):
        """
        Base model initialization.

        :param logger: Logger instance for logging.
        """
        self.logger = logger

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
    def __init__(self, model, logger, loss_fn=nn.MSELoss(), optimizer_cls=optim.Adam, lr=0.001, batch_size=32, epochs=100):
        """
        Initializes the TorchModel with the given parameters.

        :param model: PyTorch model to be trained.
        :param loss_fn: Loss function.
        :param optimizer_cls: Optimizer class.
        :param lr: Learning rate.
        :param logger: Logger instance for logging.
        """

        super().__init__(logger)

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

        if self.logger:
            self.logger.log("TorchModel initialized.", level="info")

    def train(self, X_train, y_train):
        """
        Trains the model on the provided training data.

        :param X_train: Training features.
        :param y_train: Training labels.
        """
        self.model.train()
        dataset = self.dataset(X_train.values, y_train.values)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if self.logger:
            self.logger.log("Starting model training.", level="info")

        try:
            for epoch in range(self.epochs):
                for batch, (X, y) in enumerate(dataloader):
                    X, y = X.to(self.device), y.to(self.device)

                    pred = self.model(X)
                    loss = self.loss_fn(pred, y)

                    # Backpropagation.
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if batch % 100 == 0 and self.logger:
                        self.logger.log(f"Epoch [{epoch+1}/{self.epochs}], Batch [{batch}], Loss: {loss.item():.4f}", level="debug")
                        print(f"loss: {loss:2f} [{batch} / {len(dataloader)}]")
            if self.logger:
                self.logger.log("Model training completed successfully.", level="info")

        except Exception as e:
            if self.logger:
                self.logger.log(f"Error during model training: {e}", level="error")
            raise

    def predict(self, X):
        """
        Makes predictions using the model on the provided data.

        :param X: Features to make predictions on.
        :return: Predictions.
        """
        self.model.eval()
        try:
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                predictions = self.model(X_tensor)

            if self.logger:
                self.logger.log("Predictions generated successfully.", level="info")

            return predictions.cpu().numpy()
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error during predictions: {e}", level="error")
            raise

    def evaluate(self, X_val, y_val):
        """
        Evaluate the model on a single pass of the dataloader.

        Args:
            :param X_val: Validation features.
            :param y_val: Validation labels.
        """
        dataset = self.dataset(X_val.values, y_val.values)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        num_batches = len(dataloader)
        self.model.eval()
        eval_loss = 0

        try:
            with torch.no_grad():
                for X, y in dataloader:
                    X, y = X.to(self.device), y.to(self.device)
                    pred = self.model(X)
                    eval_loss += self.loss_fn(pred, y).item()

            eval_loss /= num_batches

            if self.logger:
                self.logger.log(f"Model evaluation completed. Loss: {eval_loss:.4f}", level="info")

            return eval_loss
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error during model evaluation: {e}", level="error")
            raise
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
                self.logger.log(f"Epoch [{epoch + 1}/{self.epochs}], Batch [{batch}], Loss: {loss.item():.4f}",
                                level="debug")
                # TODO: log metric? MLflow? Custom?
                # step = batch // 100 * (epoch + 1)
                print(f"loss: {loss:2f} [{current} / {len(dataloader)}]")
    def train_evaluate(self, train_dataloader, val_dataloader, epochs=100, batch_size=32):
        # TODO: function with train_step and evaluate. End of training with EarlyStopping
        pass

class SKLearnModel(BaseModel):
    def __init__(self, model=LinearRegression(), logger=None):
        """
        Initializes the SKLearnModel with the given scikit-learn model.

        :param model: Scikit-learn model to be trained.
        :param logger: Logger instance for logging.
        """
        super().__init__(logger=logger)
        self.model = model

        if self.logger:
            self.logger.log("SKLearnModel initialized.", level="info")

    def train(self, X_train, y_train):
        """
        Trains the model on the provided training data.

        :param X_train: Training features.
        :param y_train: Training labels.
        """
        try:
            self.model.fit(X_train, y_train.values)
            if self.logger:
                self.logger.log("Model training completed successfully.", level="info")
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error during model training: {e}", level="error")
            raise

    def predict(self, X):
        """
        Makes predictions using the model on the provided data.

        :param X: Features to make predictions on.
        :return: Predictions.
        """
        try:
            predictions = self.model.predict(X)
            if self.logger:
                self.logger.log("Predictions generated successfully.", level="info")
            return predictions
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error during predictions: {e}", level="error")
            raise

    def evaluate(self, X_test, y_test, metric=mean_squared_error):
        """
        Evaluates the model on the provided test data.

        :param X_test: Test features.
        :param y_test: Test labels.
        :param metric: Metric to evaluate the model.
        :return: Evaluation metric value.
        """
        try:
            predictions = self.predict(X_test)
            eval_metric = metric(y_test.values, predictions)
            if self.logger:
                self.logger.log(f"Model evaluation completed. Metric: {eval_metric:.4f}", level="info")
            return eval_metric
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error during model evaluation: {e}", level="error")
            raise
