class BaseMemory:
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def predict(self, test_loader):
        pass
