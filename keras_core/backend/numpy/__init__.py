from contextlib import nullcontext

import numpy as np
from tensorflow import nest

from keras_core.backend.common import KerasVariable
from keras_core.backend.common import get_autocast_scope
from keras_core.backend.common import standardize_dtype
from keras_core.backend.common import standardize_shape
from keras_core.backend.common.keras_tensor import KerasTensor
from keras_core.backend.common.stateless_scope import StatelessScope
from keras_core.backend.common.stateless_scope import get_stateless_scope
from keras_core.backend.common.stateless_scope import in_stateless_scope
from keras_core.backend.numpy import math
from keras_core.backend.numpy import nn
from keras_core.backend.numpy import numpy
from keras_core.backend.numpy import random
from keras_core.utils.naming import auto_name

DYNAMIC_SHAPES_OK = False  # Dynamic shapes NG


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
        return true_fn
    return false_fn


def name_scope(name):
    # There is no need for a named context for NumPy.
    return nullcontext()


def vectorized_map(function, elements):
    return np.vectorize(function)(elements)


class Variable(KerasVariable):
    def _initialize(self, value):
        self._value = np.array(value, dtype=self._dtype)

    def assign(self, value):
        value = convert_to_tensor(value, dtype=self.dtype)
        if value.shape != self.shape:
            raise ValueError(
                "The shape of the target variable and "
                "the shape of the target value in "
                "`variable.assign(value)` must match. "
                f"Received: value.shape={value.shape}; "
                f"variable.shape={self.value.shape}"
            )
        if in_stateless_scope():
            scope = get_stateless_scope()
            scope.add_update((self, value))
        else:
            if isinstance(value, np.ndarray) and value.dtype == self.dtype:
                # Avoid a memory copy
                self._value = value
            else:
                self._value = np.array(value, dtype=self.dtype)

    @property
    def value(self):
        if in_stateless_scope():
            scope = get_stateless_scope()
            value = scope.get_current_value(self)
            if value is not None:
                return self._maybe_autocast(value)
        if self._value is None:
            # Unitialized variable. Return a placeholder.
            # This is fine because it's only ever used
            # in during shape inference with NumPy tracer objects
            # (anything else would be a bug, to be fixed.)
            return self._maybe_autocast(
                np.array(
                    self._initializer(self._shape, dtype=self._dtype),
                    dtype=self._dtype,
                )
            )
        return self._maybe_autocast(self._value)

    def numpy(self):
        return np.array(self.value)

    # Overload native accessor.
    def __array__(self):
        return self

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)


# Shape / dtype inference util
def compute_output_spec(fn, *args, **kwargs):
    with StatelessScope():

        def convert_keras_tensor_to_numpy(x):
            if isinstance(x, KerasTensor):
                return x.numpy()
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


def traceable_tensor(shape, dtype=None):
    """Create a "traceable tensor".

    That's a tensor that can be passed as input
    to a stateful backend-native function to
    create state during the trace.
    """
    shape = list(shape)
    dtype = dtype or "float32"
    return np.ones(shape, dtype=dtype)
