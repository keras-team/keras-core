"""
scatter
"""

from keras_core import backend
from keras_core.backend import KerasTensor
from keras_core.backend import any_symbolic_tensors
from keras_core.operations.operation import Operation


class Scatter(Operation):
    def call(self, x, indices, values):
        return backend.core.scatter(x, indices, values)

    def compute_output_spec(self, x, indices, values):
        return KerasTensor(x.shape, dtype=x.dtype)


def scatter(x, indices, values):
    if any_symbolic_tensors((x,)):
        return Scatter().symbolic_call(x, indices, values)
    return backend.core.scatter(x, indices, values)
