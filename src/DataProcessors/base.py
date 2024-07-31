import pandas as pd
from sklearn.model_selection import train_test_split


class BaseDataHandler:
    """
    A class for data processing (loading, preprocessing, splitting into training and test sets).
    """

    def __init__(self, data_path=None, data=None):
        assert data_path is not None or data is not None, "Please gives data or path to data"
        self.data_path = data_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        return self.data

    def preprocess_data(self):
        pass

    def split_data(self, test_size=0.2):
        train_data, test_data = train_test_split(self.data, test_size=test_size)
        return train_data, test_data
