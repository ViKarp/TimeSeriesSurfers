from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class BaseTrigger(ABC):
    @abstractmethod
    def check(self, model, true_data, predicted_data):
        """
        An abstract method to be implemented by subclasses to check if the model needs to be retrained.

        :param model: The model to be checked.
        :param true_data: The true data used to evaluate the condition.
        :param predicted_data: The predicted data used to evaluate the condition.
        :return: Boolean indicating whether retraining is needed.
        """
        pass


class PerformanceTrigger(BaseTrigger):
    def __init__(self, metric=r2_score, threshold=0.1):
        """
        Initializes the PerformanceTrigger with the given performance threshold.

        :param metric: Metric to be used for calculating the performance.
        :param threshold: Performance threshold below which retraining is triggered.
        """
        self.metric = metric
        self.threshold = threshold
        self.performance_history = []

    def check(self, model, true_data, predicted_data):
        """
        Checks if the model's performance on the data is below the threshold.

        :param model: The model to be checked.
        :param true_data: The true data used to evaluate the condition.
        :param predicted_data: The predicted data used to evaluate the condition.
        :return: Boolean indicating whether retraining is needed.
        """
        performance = self.metric(true_data, predicted_data)
        self.performance_history.append(performance)
        if len(self.performance_history) > 1:
            change = self.performance_history[-2] - self.performance_history[-1]
            if change > self.threshold:
                print("Retraining needed")
                return True
        return False


class DriftTrigger(BaseTrigger):
    def __init__(self, drift_threshold):
        """
        Initializes the DriftTrigger with the given drift threshold.

        :param drift_threshold: Threshold for detecting drift in data distribution.
        """
        self.drift_threshold = drift_threshold

    def check(self, model, true_data, predicted_data):
        """
        Checks if there is significant drift in the data compared to the data used for training.

        :param model: The model to be checked.
        :param true_data: The true data used to evaluate the condition.
        :param predicted_data: The predicted data used to evaluate the condition.
        :return: Boolean indicating whether retraining is needed.
        """
        drift_score = self.detect_drift(true_data)
        return drift_score > self.drift_threshold

    def detect_drift(self, data):
        """
        Detects drift in the data compared to the training data.
        This is a placeholder method and should be implemented to return the actual drift score.

        :param data: The data to check for drift.
        :return: Drift score indicating the degree of drift in the data.
        """
        # Here we would typically use statistical tests to detect drift
        pass


class MeanErrorTrigger(BaseTrigger):
    def __init__(self, metric=mean_squared_error, error_threshold=50):
        """
        Initializes the MeanErrorTrigger with the given error threshold.

        :param metric: Metric to be used for calculating the performance.
        :param error_threshold: Threshold for the prediction error that triggers retraining.
        """
        self.metric = metric
        self.error_threshold = error_threshold
        self.errors = []

    def check(self, model, true_data, predicted_data):
        """
        Checks whether the average error exceeds the threshold, indicating the need for retraining.

        :return: Boolean indicating whether retraining is needed.
        """
        if len(self.errors) == 0:
            return False
        error = self.metric(true_data, predicted_data)
        self.errors.append(error)
        average_error = sum(self.errors) / len(self.errors)
        return average_error > self.error_threshold
