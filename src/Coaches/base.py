from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from src.utils.plotting import plot_trigger_list, plot_results

class BaseCoach(ABC):
    def __init__(self, data_processor, memory, trigger, model, train_data, target, logger):
        """
        Initializes the BaseCoach with the given components and trains the model.

        :param data_processor: An instance of a BaseDataProcessor or similar class.
        :param memory: An instance of a BaseMemory or similar class.
        :param trigger: An instance of a BaseTrigger or similar class.
        :param model: An instance of a BaseModel or similar class.
        :param train_data: Training time_series.
        :param target: List of the names of the target variable.
        """
        self.logger = logger
        self.logger.log('Initializing BaseCoach.')
        self.data_processor = data_processor(target=target, val_size=0.2)

        self.logger.log('Initialized DataProcessor.')
        self.memory = memory(data=train_data)

        self.logger.log('Initialized Coach.')
        self.trigger = trigger(logger=self.logger)

        self.logger.log('Initialized Trigger.')
        self.model = model()

        self.logger.log('Initialized Model.')
        self.last_index = 0
        self.last_predict = None
        self.target = target

        # Train the model using the provided data
        self.logger.log('Start train first model.')
        self.train(train_data)

    def train(self, train_data):
        """
        Trains the model using the provided training data and updates the memory.

        :param train_data: data for train model as pd.DataFrame.
        """
        # Preprocess data
        self.logger.log('Preprocessing Data.')
        train_data, val_data = self.data_processor.split_data(train_data)
        # train_data, val_data is all DataFrame, not only target
        train_data_scaled, val_data_scaled = self.data_processor.preprocess_data(train_data, val_data)
        # train_data_scaled, val_data_scaled is a DataFrame with only target, index is a Timestamp
        self.logger.log('End of Preprocessing Data.')

        # Train the model
        self.logger.log('Training end evaluating model.')
        self.model.train(np.arange(len(train_data_scaled)), train_data_scaled)
        self.last_index = len(train_data_scaled) - 1

        # Evaluate the model
        eval_metric = self.model.evaluate(np.arange(self.last_index + 1, self.last_index + 1 + len(val_data_scaled)),
                                          val_data_scaled)
        self.logger.log(f"Model trained. Eval metrics:  Avg loss: {eval_metric:2f} \n")
        self.predict(val_data_scaled.index)

    def predict(self, timestamps):
        """
        Makes predictions using the model on the provided data.

        :param timestamps: Timestamps to make predictions on.
        """
        self.logger.log('Start predictions.')
        self.last_index += len(timestamps)
        self.last_predict = self.model.predict(np.arange(self.last_index + 1, self.last_index + 1 + len(timestamps)))
        self.logger.log('End predictions.')
        self.memory.load_predicted_data(
            pd.DataFrame(self.last_predict,
                         columns=self.target, index=timestamps))
        self.logger.log('Updated memory with predicted data.')

    def get_new_data(self, new_data):
        """
        Gets new data, updates the memory, and checks if retraining is needed.

        :param new_data: DataFrame containing new data.
        """
        self.logger.log('Load new values, update memory, starting check quality.')
        self.memory.load_new_data(new_data)
        self.check_quality(new_data[self.target].values)

    def check_quality(self, new_data):
        """
        Checks whether the model needs retraining based on the trigger.

        :param new_data: np.array containing new data.
        """
        if self.trigger.check(new_data, self.last_predict):
            self.logger.log('Starting refit model.')
            self.refit_model()

    def refit_model(self):
        """
        Re-trains the model using the data stored in memory.
        """
        train_data = self.memory.data
        self.train(train_data)

    def summing_up(self, dir_path):
        """
        Function to generate graphics, calculate statistics and metrics.
        :param dir_path:
        :return:
        """
        # TODO: it
        pass
