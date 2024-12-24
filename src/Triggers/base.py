from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class BaseTrigger(ABC):
    def __init__(self, logger):
        self.logger = logger
        if self.logger:
            self.logger.log("BaseTrigger initialized.", level="info")

    @abstractmethod
    def check(self, true_data, predicted_data):
        """
        An abstract method to be implemented by subclasses to check if the model needs to be retrained.

        :param true_data: The true data used to evaluate the condition.
        :param predicted_data: The predicted data used to evaluate the condition.
        :return: Boolean indicating whether retraining is needed.
        """
        pass

    @abstractmethod
    def give_results(self):
        """
        An abstract method to be implemented by subclasses to give the results of calculating metrics.
        :return: list with the results.
        """


class PerformanceTrigger(BaseTrigger):
    def __init__(self, logger, metric=r2_score, threshold: float = 0.1):
        """
        Initializes the PerformanceTrigger with the given performance threshold.
        This trigger compares recent performances and if a certain degradation occurs,
        it tells you to retrain the model.

        :param metric: Metric to be used for calculating the performance.
        :param threshold: Performance threshold below which retraining is triggered.
        """
        super().__init__(logger)
        self.metric = metric
        self.threshold = threshold
        self.performance_history = []
        if self.logger:
            self.logger.log(f"PerformanceTrigger initialized with threshold: {self.threshold}.", level="info")

    def check(self, true_data, predicted_data):
        """
        Checks if the model's performance on the data is below the threshold.

        :param true_data: The true data used to evaluate the condition.
        :param predicted_data: The predicted data used to evaluate the condition.
        :return: Boolean indicating whether retraining is needed.
        """
        try:
            performance = self.metric(true_data, predicted_data)
            self.logger.log(f"Performance: {performance}.", level="info")
            self.performance_history.append(performance)

            if len(self.performance_history) > 1:
                change = self.performance_history[-2] - self.performance_history[-1]
                self.logger.log(f"Change performance: {change}. Threshold: {self.threshold}.", level="info")
                if change > self.threshold:
                    self.logger.log("Retraining needed.", level="warning")
                    return True
            return False
        except Exception as e:
            self.logger.log(f"Error during performance check: {e}", level="error")
            raise

    def give_results(self):
        """
        Gives the performance history list.
        :return: list of performance history
        """
        return self.performance_history


class DriftTrigger(BaseTrigger):
    def __init__(self, logger, drift_threshold):
        """
        Initializes the DriftTrigger with the given drift threshold.
        This trigger compares the data and if there is a certain change in that data,
        it tells you to retrain the model.
        :param drift_threshold: Threshold for detecting drift in data distribution.
        """
        super().__init__(logger)
        self.drift_list = []
        self.drift_threshold = drift_threshold
        if self.logger:
            self.logger.log(f"DriftTrigger initialized with drift threshold: {self.drift_threshold}.", level="info")

    def check(self, true_data, predicted_data):
        """
        Checks if there is significant drift in the data compared to the data used for training.

        :param true_data: The true data used to evaluate the condition.
        :param predicted_data: The predicted data used to evaluate the condition.
        :return: Boolean indicating whether retraining is needed.
        """
        try:
            drift_score = self.detect_drift(true_data)
            self.drift_list.append(drift_score)
            self.logger.log(f"Drift data score: {drift_score}. Threshold: {self.drift_threshold}.", level="info")
            return drift_score > self.drift_threshold
        except Exception as e:
            self.logger.log(f"Error during drift check: {e}", level="error")
            raise

    def detect_drift(self, data):
        """
        Detects drift in the data compared to the training data.
        This is a placeholder method and should be implemented to return the actual drift score.

        :param data: The data to check for drift.
        :return: Drift score indicating the degree of drift in the data.
        """
        pass

    def give_results(self):
        """
        Gives the results of the computed drift data.
        :return: list of drift scores
        """
        return self.drift_list


class MeanErrorTrigger(BaseTrigger):
    def __init__(self, logger, metric=mean_squared_error, error_threshold: float = 50):
        """
        Initializes the MeanErrorTrigger with the given error threshold.
        This trigger finds the average of all errors and, at a certain level, tells you to retrain the model.

        :param metric: Metric to be used for calculating the performance.
        :param error_threshold: Threshold for the prediction error that triggers retraining.
        """
        super().__init__(logger)
        self.metric = metric
        self.error_threshold = error_threshold
        self.errors = []
        if self.logger:
            self.logger.log(f"MeanErrorTrigger initialized with error threshold: {self.error_threshold}.", level="info")

    def check(self, true_data, predicted_data):
        """
        Checks whether the average error exceeds the threshold, indicating the need for retraining.

        :return: Boolean indicating whether retraining is needed.
        """
        try:
            error = self.metric(true_data, predicted_data)
            self.logger.log(f"Error: {error}.", level="info")
            self.errors.append(error)
            average_error = sum(self.errors) / len(self.errors)
            self.logger.log(f"AVG error: {average_error}. Threshold: {self.error_threshold}.", level="info")
            return average_error > self.error_threshold
        except Exception as e:
            self.logger.log(f"Error during mean error check: {e}", level="error")
            raise

    def give_results(self):
        """
        Gives the results of the calculation of the error.
        :return: list of errors.
        """
        return self.errors
