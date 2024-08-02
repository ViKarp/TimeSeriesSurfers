from abc import ABC, abstractmethod
from src.DataStreamers.base import BaseDataStreamer
from src.Coaches.base import BaseCoach
from src.Loggers.base import BaseLogger

from src.Models.base import SKLearnModel
from src.DataProcessors.base import BaseDataProcessor
from src.Memory.base import BaseMemory
from src.Triggers.base import PerformanceTrigger


class BasePipeline(ABC):
    def __init__(self, data_streamer, coach, logger, data_path, target,
                 data_processor, memory, trigger, model_class, model, logging_file_path, results_path,
                 split_ratio: float = 0.2,
                 chunk_size: int = 1):
        """
        Initializes the BasePipeline with DataStreamer, Coach, and Logger.

        :param data_streamer: An instance of a DataStreamer or similar class.
        :param coach: An instance of a BaseCoach or similar class.
        :param logger: An instance of a Logger or similar class.
        """
        # TODO: its need to do with cfg. Class Pipeline must have a instance only Coach and DataStreamer
        # TODO: all_seed
        self.logger = logger(log_file=logging_file_path)
        self.data_streamer = data_streamer(data_path=data_path, target=target, split_ratio=split_ratio,
                                           chunk_size=chunk_size, logger=self.logger)
        self.coach = coach(data_processor=data_processor, memory=memory, trigger=trigger, model=model,
                           model_class=model_class, train_data=self.data_streamer.get_train_data(), target=target,
                           logger=self.logger)
        self.results_path = results_path

    @abstractmethod
    def run(self):
        """
        Runs the pipeline by getting chunks from DataStreamer, making predictions with Coach,
        and updating the model with new data until all chunks are processed.
        """
        pass


class ClassicPipeline(BasePipeline):
    def run(self):
        """
        Runs the pipeline by getting chunks from DataStreamer, making predictions with Coach,
        and updating the model with new data until all chunks are processed.
        """
        for chunk in self.data_streamer.get_next_chunk():
            self.logger.log(
                f"Processing new chunk of data. Times: {chunk['timestamp'].iloc[0]} - {chunk['timestamp'].iloc[-1]}")

            # Make predictions
            self.coach.predict(chunk['timestamp'].values)
            self.logger.log("Take Predictions.")

            # Get true data
            self.coach.get_new_data(chunk)
            self.logger.log("Get true data to coach.")
        self.logger.log("Finished. Generate graphics and calculate metrics.")
        self.coach.summing_up(dir_path=self.results_path)
