class BaseTrigger:
    def __init__(self, threshold):
        self.threshold = threshold
        self.performance_history = []

    def check(self, performance):
        self.performance_history.append(performance)
        if len(self.performance_history) > 1:
            change = self.performance_history[-1] - self.performance_history[-2]
            if change < self.threshold:
                print("Retraining needed")
                return True
        return False
