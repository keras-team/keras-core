"""
scatter
"""

from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.backend import KerasTensor
from keras_core.backend import any_symbolic_tensors
from keras_core.operations.operation import Operation


class Scatter(Operation):
    def call(self, indices, values, shape):
        return backend.core.scatter(indices, values, shape)

    def compute_output_spec(self, indices, values, shape):
        return KerasTensor(shape, dtype=values.dtype)


@keras_core_export("keras_core.operations.scatter")
def scatter(indices, values, shape):
    if any_symbolic_tensors((indices, values, shape)):
        return Scatter().symbolic_call(indices, values, shape)
    return backend.core.scatter(indices, values, shape)


class ScatterUpdate(Operation):
    def call(self, inputs, indices, updates):
        return backend.core.scatter_update(inputs, indices, updates)

    def compute_output_spec(self, inputs, indices, updates):
        return KerasTensor(inputs.shape, dtype=inputs.dtype)


@keras_core_export("keras_core.operations.scatter_update")
def scatter_update(inputs, indices, updates):
    if any_symbolic_tensors((inputs, indices, updates)):
        return ScatterUpdate().symbolic_call(inputs, indices, updates)
    return backend.core.scatter_update(inputs, indices, updates)


class BlockUpdate(Operation):
    def call(self, inputs, start_indices, updates):
        return backend.core.block_update(inputs, start_indices, updates)

    def compute_output_spec(self, inputs, start_indices, updates):
        return KerasTensor(inputs.shape, dtype=inputs.dtype)


@keras_core_export("keras_core.operations.block_update")
def block_update(inputs, start_indices, updates):
    if any_symbolic_tensors((inputs, start_indices, updates)):
        return BlockUpdate().symbolic_call(inputs, start_indices, updates)
    return backend.core.block_update(inputs, start_indices, updates)
