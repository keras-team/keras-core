from contextlib import nullcontext

import numpy as np
from tensorflow import nest

from keras_core.backend.common import KerasVariable
from keras_core.backend.common import standardize_dtype
from keras_core.backend.common.keras_tensor import KerasTensor
from keras_core.backend.common.stateless_scope import StatelessScope

DYNAMIC_SHAPES_OK = True

NUMPY_DTYPES = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
}


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


def convert_to_numpy(x):
    return np.array(x)


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
    # Check if elements is a batch of data
    if len(elements) > 1:
        # Apply function to each item in the batch
        result = np.stack(
            [np.apply_along_axis(function, 0, batch) for batch in elements]
        )
    else:
        # If it's a single data item, just apply the function
        result = np.apply_along_axis(function, 0, elements)
    return result


# Shape / dtype inference util
def compute_output_spec(fn, *args, **kwargs):
    with StatelessScope():

        def has_none_shape(x):
            if isinstance(x, KerasTensor):
                return None in x.shape
            return False

        none_in_shape = any(map(has_none_shape, nest.flatten((args, kwargs))))

        def convert_keras_tensor_to_numpy(x, fill_value=None):
            if isinstance(x, KerasTensor):
                shape = list(x.shape)
                if fill_value:
                    for i, e in enumerate(shape):
                        if e is None:
                            shape[i] = fill_value
                return np.empty(
                    shape=shape,
                    dtype=NUMPY_DTYPES[x.dtype],
                )
            return x

        args_1, kwargs_1 = nest.map_structure(
            lambda x: convert_keras_tensor_to_numpy(x, fill_value=83),
            (args, kwargs),
        )
        outputs_1 = fn(*args_1, **kwargs_1)

        outputs = outputs_1

        if none_in_shape:
            args_2, kwargs_2 = nest.map_structure(
                lambda x: convert_keras_tensor_to_numpy(x, fill_value=89),
                (args, kwargs),
            )
            outputs_2 = fn(*args_2, **kwargs_2)

            flat_out_1 = nest.flatten(outputs_1)
            flat_out_2 = nest.flatten(outputs_2)

            flat_out = []
            for x1, x2 in zip(flat_out_1, flat_out_2):
                shape = list(x1.shape)
                for i, e in enumerate(x2.shape):
                    if e != shape[i]:
                        shape[i] = None
                flat_out.append(KerasTensor(shape, standardize_dtype(x1.dtype)))
            outputs = nest.pack_sequence_as(outputs_1, flat_out)

        def convert_numpy_to_keras_tensor(x):
            if is_tensor(x):
                return KerasTensor(x.shape, standardize_dtype(x.dtype))
            return x

        output_spec = nest.map_structure(convert_numpy_to_keras_tensor, outputs)
    return output_spec
