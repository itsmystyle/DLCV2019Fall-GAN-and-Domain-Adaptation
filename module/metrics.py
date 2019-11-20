class Metrics:
    def __init__(self):
        self.name = "Metric Name"

    def reset(self):
        pass

    def update(self, predicts, targets):
        pass

    def get_score(self):
        pass


class Accuracy(Metrics):
    def __init__(self, threshold=0.5):
        self.name = "Acc."
        self.threshold = threshold
        self.n_correct = 0
        self.n = 1e-20

    def reset(self):
        self.n_correct = 0
        self.n = 1e-20

    def update(self, predicts, targets):
        self.n_correct += ((predicts > 0.5) == targets).sum()
        self.n += targets.shape[0]

    def get_score(self):
        return self.n_correct / self.n
