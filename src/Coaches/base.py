import os
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

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
        self.logger.log('Initializing BaseCoach.', level="info")

        self.data_processor = data_processor(target=target, val_size=0.2)
        self.logger.log('DataProcessor initialized.', level="info")

        self.memory = memory(data=train_data)
        self.logger.log('Memory initialized.', level="info")

        self.trigger = trigger(logger=self.logger)
        self.logger.log('Trigger initialized.', level="info")

        self.model = model_class(model=model, logger=self.logger)
        self.logger.log('Model initialized.', level="info")

        self.last_index = 0
        self.last_predict = None
        self.target = target

        self.logger.log('Starting initial model training.', level="info")
        self.train(train_data)

    def train(self, train_data):
        """
        Trains the model using the provided training data and updates the memory.

        :param train_data: data for train model as pd.DataFrame.
        """
        try:
            # Preprocess data
            self.logger.log('Starting data preprocessing.', level="info")
            train_data, val_data = self.data_processor.split_data(train_data)
            # train_data, val_data is all DataFrame, not only target
            train_data_scaled, val_data_scaled = self.data_processor.preprocess_data(train_data, val_data)
            # train_data_scaled, val_data_scaled is a DataFrame with only target, index is a Timestamp
            self.logger.log('Data preprocessing completed.', level="info")
            # Train the model
            self.logger.log('Training and evaluating model.', level="info")
            self.model.train(np.arange(len(train_data_scaled)).reshape(-1, 1), train_data_scaled)
            self.last_index = len(train_data_scaled) - 1
            # Evaluate the model
            eval_metric = self.model.evaluate(
                np.arange(self.last_index + 1, self.last_index + 1 + len(val_data_scaled)).reshape(-1, 1),
                val_data_scaled)

            self.logger.log(f"Model training completed. Evaluation metric: Avg loss: {eval_metric:.6f}", level="info")
            self.predict(val_data_scaled.index)
        except Exception as e:
            self.logger.log(f"Error during model training: {e}", level="error")
            raise

    def predict(self, timestamps):
        """
        Makes predictions using the model on the provided data.

        :param timestamps: Timestamps to make predictions on.
        """
        try:
            self.logger.log('Starting predictions.', level="info")
            self.last_index += len(timestamps)
            last_predict = pd.DataFrame(self.model.predict(
                np.arange(self.last_index + 1, self.last_index + 1 + len(timestamps)).reshape(-1, 1)),
                columns=self.target, index=timestamps)
            self.last_predict = self.data_processor.inverse_transform(last_predict)
            self.logger.log('Predictions completed.', level="info")
            self.memory.load_predicted_data(self.last_predict)
            self.logger.log('Predicted data loaded into memory.', level="info")
        except Exception as e:
            self.logger.log(f"Error during predictions: {e}", level="error")
            raise

    def get_new_data(self, new_data):
        """
        Gets new data, updates the memory, and checks if retraining is needed.

        :param new_data: DataFrame containing new data.
        """
        try:
            self.logger.log('Loading new data and updating memory.', level="info")
            self.memory.load_new_data(new_data)
            self.logger.log('New data loaded into memory. Checking model quality.', level="info")
            self.check_quality(new_data)
        except Exception as e:
            self.logger.log(f"Error updating memory with new data: {e}", level="error")
            raise

    def check_quality(self, new_data):
        """
        Checks whether the model needs retraining based on the trigger.

        :param new_data: pd.DataFrame containing new data.
        """
        try:
            if self.trigger.check(true_data=new_data[self.target].values, predicted_data=self.last_predict[self.target].values):
                self.logger.log('Quality check failed. Retraining model.', level="warning")
                self.refit_model()
            else:
                self.logger.log('Quality check passed. No retraining needed.', level="info")
        except Exception as e:
            self.logger.log(f"Error during quality check: {e}", level="error")
            raise

    def refit_model(self):
        """
        Re-trains the model using the data stored in memory.
        """
        try:
            self.logger.log('Starting model retraining.', level="info")
            train_data = self.memory.data
            self.train(train_data)
            self.logger.log('Model retraining completed.', level="info")
        except Exception as e:
            self.logger.log(f"Error during model retraining: {e}", level="error")
            raise

    def summing_up(self, dir_path):
        """
        Function to generate graphics, calculate statistics and metrics.
        :param dir_path: Directory path to save the output files.
        :return: None
        """
        try:
            self.logger.log(f'Creating directory {dir_path} for summary results.', level="info")
            os.makedirs(dir_path, exist_ok=True)

            # Plotting predicted vs real values
            self.logger.log('Generating real vs predicted plots.', level="info")
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
            self.logger.log('Real vs predicted plots saved.', level="info")

            self.logger.log('Generating trigger data plots.', level="info")
            trigger_data = self.trigger.give_results()
            plt.figure()
            plt.plot(trigger_data)
            plt.title('Trigger Data')
            plt.savefig(os.path.join(dir_path, 'trigger_data.png'))
            plt.close()
            self.logger.log('Trigger data plots saved.', level="info")

        except Exception as e:
            self.logger.log(f"Error during summing up: {e}", level="error")
            raise


class Last10ValuesCoaches(BaseCoach):

    def refit_model(self):
        """
        Re-trains the model using the data stored in memory.
        """
        try:
            self.logger.log('Retraining model with last 10 values.', level="info")
            train_data = self.memory.data.iloc[-10:]
            self.train(train_data)
            self.logger.log('Retraining with last 10 values completed.', level="info")
        except Exception as e:
            self.logger.log(f"Error during retraining with last 10 values: {e}", level="error")
            raise
