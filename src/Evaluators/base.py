import torch

class BaseEvaluator:
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def evaluate(self, test_loader):
        pass


class TorchEvaluator(BaseEvaluator):

    def __init__(self, model, criterion):
        super(TorchEvaluator, self).__init__(model, criterion)
        
    def evaluate(self, test_loader):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Test Loss: {test_loss / len(test_loader)}, Accuracy: {accuracy}%')
