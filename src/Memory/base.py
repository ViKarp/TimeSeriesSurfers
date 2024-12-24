from abc import ABC, abstractmethod
import pandas as pd


class BaseMemory(ABC):
    def __init__(self, data: pd.DataFrame, logger=None):
        """
        Initializes the BaseMemory with the given DataFrame.

        :param data: DataFrame containing the initial data.
        :param logger: Logger instance for logging.
        """
        self.data = data.copy()
        self.predicted_data = None
        self.logger = logger
        if self.logger:
            self.logger.log("BaseMemory initialized with data.", level="info")

    def load_new_data(self, new_data: pd.DataFrame):
        """
        Loads new correct data into the memory.

        :param new_data: DataFrame containing the new correct data.
        """
        try:
            self.data = pd.concat([self.data, new_data]).sort_index()
            if self.logger:
                self.logger.log(f"Loaded new data into memory. Total records: {len(self.data)}.", level="info")
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error while loading new data: {e}", level="error")
            raise

    def load_predicted_data(self, predicted_data: pd.DataFrame):
        """
        Loads new predicted data into the memory.

        :param predicted_data: DataFrame containing the new predicted data.
        """
        try:
            if predicted_data is None:
                self.predicted_data = predicted_data
                if self.logger:
                    self.logger.log("Initialized predicted data as None.", level="info")
            else:
                combined_data = pd.concat([self.predicted_data, predicted_data])
                self.predicted_data = combined_data[~combined_data.index.duplicated(keep='first')].sort_index()
                if self.logger:
                    self.logger.log(f"Loaded predicted data into memory. Total records: {len(self.predicted_data)}.", level="info")
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error while loading predicted data: {e}", level="error")
            raise

    def get_correct_data_by_index(self, indices):
        """
        Returns the correct data for the given indices.

        :param indices: List or array of indices.
        :return: DataFrame with the correct data for the given indices.
        """
        try:
            correct_data = self.data.iloc[indices]
            if self.logger:
                self.logger.log(f"Retrieved correct data by indices. Records: {len(correct_data)}.", level="info")
            return correct_data
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error while retrieving correct data by indices: {e}", level="error")
            raise

    def get_predicted_data_by_index(self, indices):
        """
        Returns the predicted data for the given indices.

        :param indices: List or array of indices.
        :return: DataFrame with the predicted data for the given indices.
        """
        try:
            predicted_data = self.predicted_data.iloc[indices]
            if self.logger:
                self.logger.log(f"Retrieved predicted data by indices. Records: {len(predicted_data)}.", level="info")
            return predicted_data
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error while retrieving predicted data by indices: {e}", level="error")
            raise

    def get_correct_data_by_time(self, timestamps):
        """
        Returns the correct data for the given time range.

        :param timestamps: List or array of time
        :return: DataFrame with the correct data for the given time range.
        """
        try:
            correct_data = self.data[self.data['timestamp'] == timestamps]
            if self.logger:
                self.logger.log(f"Retrieved correct data by timestamps. Records: {len(correct_data)}.", level="info")
            return correct_data
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error while retrieving correct data by timestamps: {e}", level="error")
            raise

    def get_predicted_data_by_time(self, timestamps):
        """
        Returns the predicted data for the given time range.

        :param timestamps: List or array of time
        :return: DataFrame with the predicted data for the given time range.
        """
        try:
            predicted_data = self.predicted_data[self.predicted_data['timestamp'] == timestamps]
            if self.logger:
                self.logger.log(f"Retrieved predicted data by timestamps. Records: {len(predicted_data)}.", level="info")
            return predicted_data
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error while retrieving predicted data by timestamps: {e}", level="error")
            raise
