
import pickle
import numpy as np

from .module import Module
from .loss import MeanSquaredError, BinaryCrossEntropy, CategoricalCrossEntropy
from .optimizer import GradientDescent, Adam
from .layer import Dense, Dropout
from .activation import ReLU, Sigmoid, Softmax


LOSS_MAP = {
    'MeanSquaredError': MeanSquaredError,
    'BinaryCrossEntropy': BinaryCrossEntropy,
    'CategoricalCrossEntropy': CategoricalCrossEntropy,
}

OPTIMIZER_MAP = {
    'GradientDescent': GradientDescent,
    'Adam': Adam,
}

LAYER_MAP = {
    'Dense': Dense,
    'Dropout': Dropout,
    'ReLU': ReLU,
    'Sigmoid': Sigmoid,
    'Softmax': Softmax,
}


def load_model(file_path):
    with open(file_path, 'rb') as f:
        model_data = pickle.load(f)

    class Model(Module):
        def __init__(self):
            super().__init__()

    model = Model()

    loss_type = model_data.get("loss_fn")
    if loss_type:
        loss_cls = LOSS_MAP.get(loss_type)
        model.set_loss_fn(loss_cls())

    model.set_learning_rate(model_data.get("learning_rate"))

    optimizer_type = model_data.get("optimizer")
    if optimizer_type:
        optimizer_cls = OPTIMIZER_MAP.get(optimizer_type)
        model.set_optimizer(optimizer_cls())

    layer_data = model_data.get("layer_data", {})
    for i in range(len(layer_data)):
        key = f"layer_{i}"
        layer_info = layer_data.get(key)

        layer_type = layer_info.get("type")

        if layer_type == "Dense":
            weights = np.array(layer_info.get("weights"))
            bias = np.array(layer_info.get("bias"))

            input_dim, output_dim = weights.shape
            layer = Dense(input_dim, output_dim)
            layer.W = weights
            layer.b = bias
        elif layer_type == "Dropout":
            dropout_rate = layer_info.get("dropout_rate", 0.5)
            layer = Dropout(dropout_rate)
        else:
            layer = LAYER_MAP[layer_type]()

        model.add(layer)

    return model
