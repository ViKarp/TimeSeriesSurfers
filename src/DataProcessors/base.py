import pandas as pd
import numpy as np
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


class BaseDataProcessor:
    """
    A class for data processing (loading, preprocessing, splitting into training and test sets).
    """

    def __init__(self, target: list, val_size: float = 0.2):
        """
        Initializes the BaseDataProcessor with the given parameters.

        :param target: List of the names of the target variable.
        :param val_size: Ratio to split the data into Train and Validation parts.
        """
        self.target = target
        self.val_size = val_size
        self.scaler = NoScaler()

    def split_data(self, data: pd.DataFrame):
        """
        Splits the given DataFrame into Train and Validation parts.

        :param data: DataFrame containing the data to split.
        :return: A tuple of Train and Validation DataFrames.
        """
        train_data, val_data = train_test_split(data, test_size=self.val_size, shuffle=False)
        return train_data, val_data

    def preprocess_data(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        """
        Preprocesses the data by fitting the scaler on the Train data and transforming both Train and Validation data.

        :param train_data: DataFrame containing the Train data.
        :param val_data: DataFrame containing the Validation data.
        :return: A tuple of preprocessed Train and Validation DataFrames.
        """
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

        return train_data_scaled, val_data_scaled

    def inverse_transform(self, data):
        """
        Inverse transforms the given data using the previously fitted scaler.

        :param data: DataFrame or Series containing the data to inverse transform.
        :return: A DataFrame or Series with the inverse transformed data.
        """
        data_inverse_transformed = data.copy()
        if len(self.target) == 1:
            data_inverse_transformed = data_inverse_transformed[self.target[0]].values.reshape(-1, 1)
            data_inverse_transformed = self.scaler.inverse_transform(data_inverse_transformed)
            data_inverse_transformed = pd.DataFrame(data_inverse_transformed.flatten(), index=data['timestamp'],
                                                    columns=self.target)
        else:
            data_inverse_transformed.iloc[:, :] = self.scaler.inverse_transform(data_inverse_transformed)

        return data_inverse_transformed


class StandardScalerDataProcessor(BaseDataProcessor):
    def __init__(self, target: list, val_size: float = 0.2):
        """
        Initializes the StandardScalerDataProcessor with the given parameters.

        :param target: List of the names of the target variable.
        :param val_size: Ratio to split the data into Train and Validation parts.
        """
        super(BaseDataProcessor, self).__init__(target, val_size)
        self.scaler = StandardScaler()
