from abc import ABC, abstractmethod
import pandas as pd


class BaseDataStreamer(ABC):
    def __init__(self, data_path: str, target: str, split_ratio: float = 0.2, chunk_size: int = 1):
        """
        Class for separating data into past and future values

        :param data_path: Path to the CSV file containing the time series data.
        :param target: Name of the target variable.
        :param split_ratio: Ratio to split the data into Train and Online parts.
        :param chunk_size: Size of the chunks to be returned by get_next_chunk.
        """
        self.data_path = data_path
        self.target = target
        self.split_ratio = split_ratio
        self.chunk_size = chunk_size

        # Load and sort data
        self.data = pd.read_csv(self.data_path, parse_dates=['timestamp'])
        self.data = self.data[['timestamp', target]]
        self.data = self.data.sort_values('timestamp')

        # Split data into Train and Online parts
        split_index = int(len(self.data) * self.split_ratio)
        self._train_data = self.data.iloc[:split_index]
        self._online_data = self.data.iloc[split_index:]

    def get_train_data(self):
        """
        Returns the Train part of the data.

        :return: Train data as a pandas DataFrame.
        """
        return self._train_data

    def get_next_chunk(self):
        """
        Returns the next chunk of data from the Online part.

        :return: Generator of next chunk of data as a pandas DataFrame.
        """
        for start in range(0, (len(self._online_data)//self.chunk_size) * self.chunk_size, self.chunk_size):
            yield self._online_data.iloc[start:start + self.chunk_size]
