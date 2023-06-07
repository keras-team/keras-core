from contextlib import nullcontext

import numpy as np
from tensorflow import nest

from keras_core.backend.common import KerasVariable
from keras_core.backend.common import standardize_dtype
from keras_core.backend.common.keras_tensor import KerasTensor
from keras_core.backend.common.stateless_scope import StatelessScope

DYNAMIC_SHAPES_OK = False


class Variable(KerasVariable):
    def _initialize(self, value):
        self._value = np.array(value, dtype=self._dtype)

    def _direct_assign(self, value):
        self._value = np.array(value, dtype=self._dtype)

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)

    # Overload native accessor.
    def __array__(self):
        return self.value


def convert_to_tensor(x, dtype=None):
    if dtype is not None:
        dtype = standardize_dtype(dtype)
    if isinstance(x, Variable):
        if dtype and dtype != x.dtype:
            return x.value.astype(dtype)
        return x.value
    return np.array(x, dtype=dtype)


def is_tensor(x):
    if isinstance(x, np.ndarray):
        return True
    return False


def shape(x):
    # This will work as long as we disallow
    # dynamic shapes in NumPy.
    return x.shape


def cast(x, dtype):
    return convert_to_tensor(x, dtype=dtype)


def cond(pred, true_fn, false_fn):
    if pred:
        return true_fn()
    return false_fn()


def name_scope(name):
    # There is no need for a named context for NumPy.
    return nullcontext()


def vectorized_map(function, elements):
    return np.vectorize(function)(elements)


# Shape / dtype inference util
def compute_output_spec(fn, *args, **kwargs):
    with StatelessScope():

        def convert_keras_tensor_to_numpy(x):
            if isinstance(x, KerasTensor):
                return np.ones(x.shape, dtype=x.dtype)
            return x

        args, kwargs = nest.map_structure(
            convert_keras_tensor_to_numpy, (args, kwargs)
        )
        np_out = fn(*args, **kwargs)

        def convert_numpy_to_keras_tensor(x):
            if isinstance(x, np.ndarray):
                return KerasTensor(x.shape, x.dtype)
            return x

        return nest.map_structure(convert_numpy_to_keras_tensor, np_out)
