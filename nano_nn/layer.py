
import numpy as np


class Layer:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, out_gradient):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, in_features, out_features):
        limit = np.sqrt(6 / (in_features + out_features))
        self.W = np.random.uniform(low=-limit, high=limit, size=(in_features, out_features))
        self.b = np.zeros(out_features)

    def forward(self, x):
        self.input = x
        return x @ self.W + self.b

    def backward(self, out_gradient):
        self.grad_weights = self.input.T @ out_gradient
        self.grad_biases = np.sum(out_gradient, axis=0, keepdims=True)

        in_gradient = out_gradient @ self.W.T

        return in_gradient


class Dropout(Layer):
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.is_training = True

    def forward(self, x):
        if self.is_training:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape)
            return x * self.mask / (1 - self.dropout_rate)

        return x

    def backward(self, out_gradient):
        return out_gradient * self.mask
