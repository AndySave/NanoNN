
from .layer import Dropout
from .optimizer import GradientDescent
import pickle


class Module:
    def __init__(self):
        self.layers = []
        self.loss_fn = None
        self.learning_rate = None
        self.optimizer = GradientDescent()

    def add(self, layer):
        self.layers.append(layer)

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        self.optimizer.update_learning_rate(learning_rate)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.optimizer.update_learning_rate(self.learning_rate)

    def train(self):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.is_training = True

    def eval(self):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.is_training = False

    def forward(self, x, target=None):
        for layer in self.layers:
            x = layer.forward(x)

        if self.loss_fn and target is not None:
            loss = self.loss_fn.forward(x, target)
            return x, loss

        return x

    def backward(self):
        if not self.loss_fn:
            raise ValueError('No loss function set. Set using method set_loss_fn.')
        if not self.learning_rate:
            raise ValueError('No learning rate set. Set using method set_learning_rate.')

        loss_gradient = self.loss_fn.backward()
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient)
            if hasattr(layer, 'W'):
                self.optimizer.update(layer)

    def save_model(self, file_path):
        model_data = {'loss_fn': self.loss_fn.type, 'learning_rate': self.learning_rate, 'optimizer': self.optimizer.type}

        layer_data = {}
        for i, layer in enumerate(self.layers):
            layer_type = layer.type
            layer_data[f'layer_{i}'] = {'type': layer_type}

            if hasattr(layer, 'get_weights'):
                layer_data[f'layer_{i}']['weights'] = layer.get_weights()
                layer_data[f'layer_{i}']['bias'] = layer.get_bias()
            elif layer_type == 'Dropout':
                layer_data[f'layer_{i}']['dropout_rate'] = layer.dropout_rate

        model_data['layer_data'] = layer_data

        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
