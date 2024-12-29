import os
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import plotly.graph_objects as go
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

    def summing_up(self, dir_path: str) -> None:
        """
        Функция для построения интерактивных графиков (Real vs Predicted и Trigger Data),
        а также для сохранения результатов в указанный каталог.

        :param dir_path: Путь к каталогу для сохранения файлов.
        :return: None
        """
        try:
            self.logger.log(f'Creating directory {dir_path} for summary results.', level="info")
            os.makedirs(dir_path, exist_ok=True)

            # Подготовка данных для графика Real vs Predicted
            self.logger.log('Generating real vs predicted interactive plots using Plotly.', level="info")
            real_data = pd.Series(
                self.memory.data[self.target].values.flatten(),
                index=self.memory.data['timestamp'].values.flatten(),
                name=self.target[0]
            )
            predicted_data = self.memory.predicted_data[self.target]

            # Синхронизация индексов реальных и предсказанных данных
            common_indices = real_data.index.intersection(predicted_data.index)
            aligned_real = real_data.loc[common_indices]
            aligned_predicted = predicted_data.loc[common_indices]

            # Создание интерактивного графика с помощью Plotly
            fig = go.Figure()

            aligned_predicted = aligned_predicted.squeeze()
            # Линия реальных данных
            fig.add_trace(
                go.Scatter(
                    x=aligned_real.index,
                    y=aligned_real.values,
                    mode='lines+markers',
                    name='Real',
                    line=dict(color='blue')
                )
            )

            # Линия предсказанных данных
            fig.add_trace(
                go.Scatter(
                    x=aligned_predicted.index,
                    y=aligned_predicted.values,
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='red', dash='dash')
                )
            )

            # Настройка оформления
            fig.update_layout(
                title='Real vs Predicted Values',
                xaxis_title='Time',
                yaxis_title='Values'
            )

            # Сохраняем интерактивный график в HTML
            real_vs_predicted_file = os.path.join(dir_path, 'predicted_vs_real.html')
            fig.write_html(real_vs_predicted_file, auto_open=False)
            self.logger.log(f'Real vs predicted plot saved to {real_vs_predicted_file}', level="info")

            # Построение интерактивного графика для Trigger Data
            self.logger.log('Generating trigger data interactive plots using Plotly.', level="info")
            trigger_data = self.trigger.give_results()
            if isinstance(trigger_data, pd.Series):
                # Если trigger_data - это Series
                x_values = trigger_data.index
                y_values = trigger_data.values
            else:
                # Иначе предполагаем, что trigger_data это массив (numpy.ndarray)
                # или просто список. Тогда генерируем индекс как список:
                x_values = list(range(len(trigger_data)))
                y_values = trigger_data

            fig_trigger = go.Figure()
            fig_trigger.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='lines',
                    name='Trigger Data',
                    line=dict(color='green')
                )
            )

            fig_trigger.update_layout(title='Trigger Data')

            trigger_file = os.path.join(dir_path, 'trigger_data.html')
            fig_trigger.write_html(trigger_file, auto_open=False)
            self.logger.log(f'Trigger data plot saved to {trigger_file}', level="info")

            # Save metrics as a table in txt file
            self.logger.log('Calculating metrics.')
            metrics = {
                'MSE': mean_squared_error(aligned_real.values, aligned_predicted.values),
                'MAE': mean_absolute_error(aligned_real.values, aligned_predicted.values),
                'R2': r2_score(aligned_real.values, aligned_predicted.values),
                'MAPE': mean_absolute_percentage_error(aligned_real.values, aligned_predicted.values)
            }

            metrics_table_path = os.path.join(dir_path, 'metrics_table.txt')
            with open(metrics_table_path, 'w') as f:
                f.write("Metric\tValue\n")
                f.write("-----------------------------------\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}\t{value:.6f}\n")
            self.logger.log(f'Metrics table saved to {metrics_table_path}.')

            # Create an image of the metrics table
            self.logger.log('Generating metrics table image.')

            # Prepare data for the table
            data = {
                "Metric": list(metrics.keys()),
                "Value": [f"{v:.6f}" for v in metrics.values()]
            }
            df = pd.DataFrame(data)

            # Create the table image
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.auto_set_column_width(col=list(range(len(df.columns))))

            metrics_image_path = os.path.join(dir_path, 'metrics_table.png')
            plt.savefig(metrics_image_path, bbox_inches='tight', dpi=300)
            plt.close()
            self.logger.log(f'Metrics table image saved to {metrics_image_path}.')


        except Exception as e:
            error_message = f"An error occurred while generating interactive plots: {str(e)}"
            self.logger.log(error_message, level="error")
            raise

        # TODO: inconsistent numbers of samples
class Last10ValuesCoaches(BaseCoach):

    def refit_model(self):
        """
        Re-trains the model using the data stored in memory.
        """
        train_data = self.memory.data.iloc[-10:]
        self.train(train_data)
