import pandas as pd


class BaseDataStreamer:
    def __init__(self, data_path, chunk_size):
        self.data = pd.read_csv(data_path)
        self.chunk_size = chunk_size
        self.index = 0

    def get_next_chunk(self):
        if self.index < len(self.data):
            end_index = min(self.index + self.chunk_size, len(self.data))
            chunk = self.data.iloc[self.index:end_index]
            self.index = end_index
            return chunk
        else:
            return None
