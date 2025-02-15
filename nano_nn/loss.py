
import numpy as np


class Loss:
    def forward(self, predicted, target):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class MeanSquaredError(Loss):
    def forward(self, predicted, target):
        self.predicted = predicted
        self.target = target.reshape(-1, 1)
        return np.mean((predicted - target) ** 2) / 2

    def backward(self):
        return (self.predicted - self.target) / self.target.size


class BinaryCrossEntropy(Loss):
    def forward(self, predicted, target):
        self.predicted = np.clip(predicted, 1e-15, 1 - 1e-15)
        self.target = target.reshape(-1, 1)
        return -np.mean(self.target * np.log(self.predicted) + (1 - self.target) * np.log(1 - self.predicted))

    def backward(self):
        grad = (self.predicted - self.target) / (self.predicted * (1 - self.predicted))
        return grad / self.target.size


class CategoricalCrossEntropy(Loss):
    def forward(self, predicted, target):
        self.predicted = np.clip(predicted, 1e-15, 1 - 1e-15)
        self.target = target
        return -np.mean(np.sum(self.target * np.log(self.predicted), axis=1))

    def backward(self):
        grad = self.predicted - self.target
        return grad / self.target.shape[0]
