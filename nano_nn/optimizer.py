
import numpy as np


class Optimizer:
    def update_learning_rate(self, learning_rate):
        raise NotImplementedError

    def update(self, layer):
        raise NotImplementedError


class GradientDescent(Optimizer):
    def __init__(self):
        self.type = 'GradientDescent'
        self.learning_rate = None

    def update_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, layer):
        layer.W -= self.learning_rate * layer.grad_weights
        layer.b -= self.learning_rate * layer.grad_biases.reshape(-1)


class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.type = 'Adam'
        self.learning_rate = None
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def update_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, layer):
        if not hasattr(layer, 'm'):
            layer.m = np.zeros_like(layer.W)
            layer.v = np.zeros_like(layer.W)
            layer.m_b = np.zeros_like(layer.b)
            layer.v_b = np.zeros_like(layer.b)

        self.t += 1

        layer.m = self.beta1 * layer.m + (1 - self.beta1) * layer.grad_weights
        layer.v = self.beta2 * layer.v + (1 - self.beta2) * (layer.grad_weights ** 2)

        m_hat = layer.m / (1 - self.beta1 ** self.t)
        v_hat = layer.v / (1 - self.beta2 ** self.t)

        layer.m_b = self.beta1 * layer.m_b + (1 - self.beta1) * layer.grad_biases
        layer.v_b = self.beta2 * layer.v_b + (1 - self.beta2) * (layer.grad_biases ** 2)

        m_hat_b = layer.m_b / (1 - self.beta1 ** self.t)
        v_hat_b = layer.v_b / (1 - self.beta2 ** self.t)

        layer.W -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        layer.b -= (self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)).squeeze()
