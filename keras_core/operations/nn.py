"""
relu
relu6
sigmoid
softplus
softsign
silu
swish
log_sigmoid
leaky_relu
hard_sigmoid
elu
selu
gelu
softmax
log_softmax

max_pooling
average_pooling
conv
depthwise_conv
separable_conv
conv_transpose

ctc ??
"""

from keras_core import backend
from keras_core.backend import KerasTensor
from keras_core.backend import any_symbolic_tensors
from keras_core.operations.operation import Operation


class Relu(Operation):
    def call(self, x):
        return backend.nn.relu(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, x.dtype)


def relu(x):
    if any_symbolic_tensors((x,)):
        return Relu().symbolic_call(x)
    return backend.nn.relu(x)


class Relu6(Operation):
    def call(self, x):
        return backend.nn.relu6(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, x.dtype)


def relu6(x):
    if any_symbolic_tensors((x,)):
        return Relu6().symbolic_call(x)
    return backend.nn.relu6(x)


class Sigmoid(Operation):
    def call(self, x):
        return backend.nn.sigmoid(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def sigmoid(x):
    if any_symbolic_tensors((x,)):
        return Sigmoid().symbolic_call(x)
    return backend.nn.sigmoid(x)


class Softplus(Operation):
    def call(self, x):
        return backend.nn.softplus(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def softplus(x):
    if any_symbolic_tensors((x,)):
        return Softplus().symbolic_call(x)
    return backend.nn.softplus(x)


class Softsign(Operation):
    def call(self, x):
        return backend.nn.softsign(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def softsign(x):
    if any_symbolic_tensors((x,)):
        return Softsign().symbolic_call(x)
    return backend.nn.softsign(x)


class Silu(Operation):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def call(self, x):
        return backend.nn.silu(x, beta=self.beta)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)
