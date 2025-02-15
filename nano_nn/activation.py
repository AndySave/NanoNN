
import numpy as np


class Activation:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, out_gradient):
        raise NotImplementedError


class ReLU(Activation):
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, out_gradient):
        in_gradient = out_gradient * (self.input > 0)
        return in_gradient


class Sigmoid(Activation):
    def forward(self, x):
        x = np.clip(x, -500, 500)
        self.sigmoid_result = 1 / (1 + np.exp(-x))
        return self.sigmoid_result

    def backward(self, out_gradient):
        in_gradient = out_gradient * self.sigmoid_result * (1 - self.sigmoid_result)
        return in_gradient


class Softmax(Activation):
    def forward(self, x):
        exp_z = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.softmax_result = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.softmax_result

    def backward(self, out_gradient):
        return out_gradient
