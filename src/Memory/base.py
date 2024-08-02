from abc import ABC, abstractmethod
import pandas as pd


class BaseMemory(ABC):
    def __init__(self, data: pd.DataFrame):
        """
        Initializes the BaseMemory with the given DataFrame.

        :param data: DataFrame containing the initial data.
        """
        self.data = data.copy()
        self.predicted_data = None

    def load_new_data(self, new_data: pd.DataFrame):
        """
        Loads new correct data into the memory.

        :param new_data: DataFrame containing the new correct data.
        """
        self.data = pd.concat([self.data, new_data]).sort_index()

    def load_predicted_data(self, predicted_data: pd.DataFrame):
        """
        Loads new predicted data into the memory.

        :param predicted_data: DataFrame containing the new predicted data.
        """
        # Check for duplicate indices and add only new data
        if predicted_data is None:
            self.predicted_data = predicted_data
        else:
            combined_data = pd.concat([self.predicted_data, predicted_data])
            self.predicted_data = combined_data[~combined_data.index.duplicated(keep='first')].sort_index()

    def get_correct_data_by_index(self, indices):
        """
        Returns the correct data for the given indices.

        :param indices: List or array of indices.
        :return: DataFrame with the correct data for the given indices.
        """
        return self.data.iloc[indices]

    def get_predicted_data_by_index(self, indices):
        """
        Returns the predicted data for the given indices.

        :param indices: List or array of indices.
        :return: DataFrame with the predicted data for the given indices.
        """
        return self.predicted_data.iloc[indices]

    def get_correct_data_by_time(self, timestamps):
        """
        Returns the correct data for the given time range.

        :param timestamps: List or array of time
        :return: DataFrame with the correct data for the given time range.
        """
        return self.data[self.data['timestamp'] == timestamps]

    def get_predicted_data_by_time(self, timestamps):
        """
        Returns the predicted data for the given time range.

        :param timestamps: List or array of time
        :return: DataFrame with the predicted data for the given time range.
        """
        return self.predicted_data[self.predicted_data['timestamp'] == timestamps]
