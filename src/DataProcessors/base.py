from abc import ABC
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class NoScaler(TransformerMixin, BaseEstimator):
    def __init__(self):
        super(TransformerMixin, self).__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


class BaseDataProcessor(ABC):
    """
    A class for data processing (loading, preprocessing, splitting into training and test sets).
    """

    def __init__(self, target: list, val_size: float = 0.2, logger=None):
        """
        Initializes the BaseDataProcessor with the given parameters.

        :param target: List of the names of the target variable.
        :param val_size: Ratio to split the data into Train and Validation parts.
        :param logger: Logger instance for logging.
        """
        self.target = target
        self.val_size = val_size
        self.scaler = NoScaler()
        self.logger = logger
        if self.logger:
            self.logger.log("Initialized BaseDataProcessor.", level="info")

    def split_data(self, data: pd.DataFrame):
        """
        Splits the given DataFrame into Train and Validation parts.

        :param data: DataFrame containing the data to split. cols: ['timestamp', target]
        :return: A tuple of Train and Validation DataFrames.
        """
        try:
            train_data, val_data = train_test_split(data, test_size=self.val_size, shuffle=False)
            if self.logger:
                self.logger.log("Data successfully split into train and validation sets.", level="info")
            return train_data, val_data
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error during data split: {e}", level="error")
            raise

    def preprocess_data(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        """
        Preprocesses the data by fitting the scaler on the Train data and transforming both Train and Validation data.

        :param train_data: DataFrame containing the Train data.
        :param val_data: DataFrame containing the Validation data.
        :return: A tuple of preprocessed Train and Validation DataFrames with one column and index is timestamp.
        """
        try:
            if self.logger:
                self.logger.log("Starting data preprocessing.", level="info")

            train_data_scaled = train_data.copy()
            val_data_scaled = val_data.copy()

            if len(self.target) == 1:
                train_data_scaled = train_data_scaled[self.target[0]].values.reshape(-1, 1)
                val_data_scaled = val_data_scaled[self.target[0]].values.reshape(-1, 1)

                self.scaler.fit(train_data_scaled)
                train_data_scaled = self.scaler.transform(train_data_scaled)
                val_data_scaled = self.scaler.transform(val_data_scaled)

                train_data_scaled = pd.DataFrame(train_data_scaled.flatten(), index=train_data['timestamp'],
                                                 columns=self.target)
                val_data_scaled = pd.DataFrame(val_data_scaled.flatten(), index=val_data['timestamp'], columns=self.target)
            else:
                self.scaler.fit(train_data_scaled)
                train_data_scaled.iloc[:, :] = self.scaler.transform(train_data_scaled)
                val_data_scaled.iloc[:, :] = self.scaler.transform(val_data_scaled)

            if self.logger:
                self.logger.log("Data preprocessing completed successfully.", level="info")

            return train_data_scaled, val_data_scaled
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error during data preprocessing: {e}", level="error")
            raise

    def inverse_transform(self, data):
        """
        Inverse transforms the given data using the previously fitted scaler.

        :param data: DataFrame or Series containing the data to inverse transform.
        :return: A DataFrame or Series with the inverse transformed data.
        """
        try:
            if self.logger:
                self.logger.log("Starting inverse transformation.", level="info")

            data_inverse_transformed = data.copy()
            if len(self.target) == 1:
                data_inverse_transformed = data_inverse_transformed[self.target[0]].values.reshape(-1, 1)
                data_inverse_transformed = self.scaler.inverse_transform(data_inverse_transformed)
                data_inverse_transformed = pd.DataFrame(data_inverse_transformed.flatten(), index=data.index,
                                                        columns=self.target)
            else:
                data_inverse_transformed.iloc[:, :] = self.scaler.inverse_transform(data_inverse_transformed)

            if self.logger:
                self.logger.log("Inverse transformation completed successfully.", level="info")

            return data_inverse_transformed
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error during inverse transformation: {e}", level="error")
            raise


class StandardScalerDataProcessor(BaseDataProcessor):
    def __init__(self, target: list, val_size: float = 0.2, logger=None):
        """
        Initializes the StandardScalerDataProcessor with the given parameters.

        :param target: List of the names of the target variable.
        :param val_size: Ratio to split the data into Train and Validation parts.
        :param logger: Logger instance for logging.
        """
        super(BaseDataProcessor, self).__init__()
        self.target = target
        self.val_size = val_size
        self.scaler = StandardScaler()
        self.logger = logger
        if self.logger:
            self.logger.log("Initialized StandardScalerDataProcessor.", level="info")
