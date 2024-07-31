import torch
from torch.utils.data import DataLoader, TensorDataset


class BasePipeline:
    def __init__(self):
        pass

    def run(self):
        pass


class TorchPipeline(BasePipeline):
    def __init__(self, data_handler, model, trainer, evaluator, logger):
        super(TorchPipeline, self).__init__()
        self.data_handler = data_handler
        self.model = model
        self.trainer = trainer
        self.evaluator = evaluator
        self.logger = logger

    def run(self, batch_size=32, epochs=10):
        self.logger.log("Loading data...")
        data = self.data_handler.load_data()
        self.data_handler.preprocess_data()
        train_data, test_data = self.data_handler.split_data()

        train_loader = DataLoader(TensorDataset(torch.tensor(train_data.iloc[:, :-1].values, dtype=torch.float32),
                                                torch.tensor(train_data.iloc[:, -1].values, dtype=torch.long)),
                                  batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(torch.tensor(test_data.iloc[:, :-1].values, dtype=torch.float32),
                                               torch.tensor(test_data.iloc[:, -1].values, dtype=torch.long)),
                                 batch_size=batch_size, shuffle=False)

        self.logger.log("Starting training...")
        self.trainer.train(train_loader, epochs)

        self.logger.log("Evaluating model...")
        self.evaluator.evaluate(test_loader)
