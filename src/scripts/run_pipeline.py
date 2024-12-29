from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from src.Pipelines.base import ClassicPipeline

from src.DataStreamers.base import BaseDataStreamer
from src.Coaches.base import Last10ValuesCoaches
from src.Loggers.base import BaseLogger

from src.Models.base import SKLearnModel
from src.DataProcessors.base import BaseDataProcessor, StandardScalerDataProcessor
from src.Memory.base import BaseMemory
from src.Triggers.base import PerformanceTrigger

from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    data_path = '../../data/raw/BTCUSDT_very_small.csv'
    target = ['close']
    logging_file_path = '../../reports/logs/log.txt'
    results_path = '../../reports/results/001/'
    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    pipeline = ClassicPipeline(BaseDataStreamer, Last10ValuesCoaches, BaseLogger, data_path, target,
                               StandardScalerDataProcessor, BaseMemory, PerformanceTrigger, SKLearnModel, gaussian_process,
                               logging_file_path, results_path, split_ratio=0.2, chunk_size=10)
    pipeline.run()
