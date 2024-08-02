import os
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, \
    root_mean_squared_error

from src.utils.plotting import plot_trigger_list, plot_results


class BaseCoach(ABC):
    def __init__(self, data_processor, memory, trigger, model_class, model, train_data, target, logger):
        """
        Initializes the BaseCoach with the given components and trains the model.

        :param data_processor: An instance of a BaseDataProcessor or similar class.
        :param memory: An instance of a BaseMemory or similar class.
        :param trigger: An instance of a BaseTrigger or similar class.
        :param model_class: An instance of a BaseModel or similar class.
        :param model: An instance of a model.
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
        self.model = model_class(model=model)

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
        self.model.train(np.arange(len(train_data_scaled)).reshape(-1, 1), train_data_scaled)
        self.last_index = len(train_data_scaled) - 1

        # Evaluate the model
        eval_metric = self.model.evaluate(
            np.arange(self.last_index + 1, self.last_index + 1 + len(val_data_scaled)).reshape(-1, 1),
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
        last_predict = pd.DataFrame(self.model.predict(
            np.arange(self.last_index + 1, self.last_index + 1 + len(timestamps)).reshape(-1, 1)),
            columns=self.target, index=timestamps)
        self.last_predict = self.data_processor.inverse_transform(last_predict)
        self.logger.log('End predictions.')
        self.memory.load_predicted_data(self.last_predict)
        self.logger.log('Updated memory with predicted data.')

    def get_new_data(self, new_data):
        """
        Gets new data, updates the memory, and checks if retraining is needed.

        :param new_data: DataFrame containing new data.
        """
        self.logger.log('Load new values, update memory, starting check quality.')
        self.memory.load_new_data(new_data)
        self.check_quality(new_data)

    def check_quality(self, new_data):
        """
        Checks whether the model needs retraining based on the trigger.

        :param new_data: pd.DataFrame containing new data.
        """
        if self.trigger.check(true_data=new_data[self.target].values, predicted_data=self.last_predict[self.target].values):
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
        :param dir_path: Directory path to save the output files.
        :return: None
        """
        os.makedirs(dir_path, exist_ok=True)

        # Plotting predicted vs real values
        fig, ax = plt.subplots(figsize=(14, 9))
        real_data = pd.Series(self.memory.data[self.target].values.flatten(),
                              index=self.memory.data['timestamp'].values.flatten(),
                              name=self.target[0])
        predicted_data = self.memory.predicted_data[self.target]

        # Align the indices of real and predicted data
        common_indices = real_data.index.intersection(predicted_data.index)
        aligned_real = real_data.loc[common_indices]
        aligned_predicted = predicted_data.loc[common_indices]

        ax.plot(aligned_real.index, aligned_real.values, label='Real', color='blue')
        ax.plot(aligned_predicted.index, aligned_predicted.values, label='Predicted', color='red',
                linestyle='dashed')

        ax.set_xlabel('Time')
        ax.set_ylabel('Values')
        ax.legend()
        plt.title('Real vs Predicted Values')
        plt.savefig(os.path.join(dir_path, 'predicted_vs_real.png'))
        plt.close()

        trigger_data = self.trigger.give_results()
        plt.figure()
        plt.plot(trigger_data)
        plt.title('Trigger Data')
        plt.savefig(os.path.join(dir_path, 'trigger_data.png'))
        plt.close()

        # Calculate and save metrics
        # metrics = {
        #     'MSE': mean_squared_error(aligned_real.values, aligned_predicted.values),
        #     'MAE': mean_absolute_error(aligned_real.values, aligned_predicted.values),
        #     'R2': r2_score(aligned_real.values, aligned_predicted.values),
        #     'MAPE': mean_absolute_percentage_error(aligned_real.values, aligned_predicted.values),
        #     'RMSE': root_mean_squared_error(aligned_real.values, aligned_predicted.values)
        # }
        #
        # with open(os.path.join(dir_path, 'metrics.txt'), 'w') as f:
        #     for metric, value in metrics.items():
        #         f.write(f"{metric}: {value}\n")
        # TODO: inconsistent numbers of samples


class Last10ValuesCoaches(BaseCoach):

    def refit_model(self):
        """
        Re-trains the model using the data stored in memory.
        """
        train_data = self.memory.data.iloc[-10:]
        self.train(train_data)
