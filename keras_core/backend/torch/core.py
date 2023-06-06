import contextlib

import numpy as np
import torch
from tensorflow import nest

from keras_core.backend.common import KerasVariable
from keras_core.backend.common import global_state
from keras_core.backend.common import standardize_dtype
from keras_core.backend.common.keras_tensor import KerasTensor
from keras_core.backend.common.stateless_scope import StatelessScope

DYNAMIC_SHAPES_OK = True


TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "uint8": torch.uint8,
    "uint32": torch.int64,  # TODO: Torch doesn't have `uint32` dtype.
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bfloat16": torch.bfloat16,
    "bool": torch.bool,
}


@contextlib.contextmanager
def device_scope(device):
    previous_device = global_state.get_global_attribute("torch_device", None)
    global_state.set_global_attribute("torch_device", device)
    try:
        yield
    finally:
        global_state.set_global_attribute("torch_device", previous_device)


def get_device():
    device = global_state.get_global_attribute("torch_device", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def to_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype = standardize_dtype(dtype)
    dtype = TORCH_DTYPES.get(dtype, None)
    if dtype is None:
        raise ValueError(f"Unsupported dtype for PyTorch: {dtype}")
    return dtype


class Variable(KerasVariable):
    def _initialize(self, value):
        self._value = torch.nn.Parameter(
            convert_to_tensor(value, dtype=self._dtype),
            requires_grad=self.trainable,
        )

    def _direct_assign(self, value):
        with torch.no_grad():
            self.value.copy_(value)

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)

    # Overload native accessor.
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        args = [
            arg.value if isinstance(arg, KerasVariable) else arg for arg in args
        ]
        if kwargs is None:
            kwargs = {}
        kwargs = {
            key: value.value if isinstance(value, KerasVariable) else value
            for key, value in kwargs.items()
        }
        return func(*args, **kwargs)

    def __array__(self, dtype=None):
        if self.value.requires_grad:
            return self.value.detach().__array__(dtype)
        return super().__array__(dtype)

    @property
    def value(self):
        value = super().value
        # Create and use a symbolic tensor stub in symbolic calls.
        if get_device() == "meta" and value.device != "meta":
            return torch.randn(
                size=value.shape,
                dtype=value.dtype,
                device="meta",
            )
        return value


def convert_to_tensor(x, dtype=None):
    # TODO: Need to address device placement arg of `as_tensor`
    dtype = to_torch_dtype(dtype or getattr(x, "dtype", None))
    if isinstance(x, Variable):
        x = x.value
    if is_tensor(x):
        if dtype and dtype != x.dtype:
            return x.to(dtype)
        return x

    # Convert to np first in case of any non-numpy, numpy-compatible array.
    x = np.array(x)
    return torch.as_tensor(x, dtype=dtype, device=get_device())


def convert_to_numpy(x):
    if is_tensor(x) and x.requires_grad:
        return x.detach().numpy()
    return np.array(x)


def is_tensor(x):
    return torch.is_tensor(x)


def shape(x):
    return x.shape


def cast(x, dtype):
    dtype = to_torch_dtype(dtype)
    if isinstance(x, KerasVariable):
        x = x.value
    if is_tensor(x):
        return x.to(dtype)
    return convert_to_tensor(x, dtype)


def name_scope(name):
    return contextlib.nullcontext()


# Shape / dtype inference util
def compute_output_spec(fn, *args, **kwargs):
    with StatelessScope():

        def has_none_shape(x):
            if isinstance(x, KerasTensor):
                return None in x.shape
            return False

        none_in_shape = any(map(has_none_shape, nest.flatten((args, kwargs))))

        def convert_keras_tensor_to_torch(x, fill_value=None):
            if isinstance(x, KerasTensor):
                shape = list(x.shape)
                if fill_value:
                    for i, e in enumerate(shape):
                        if e is None:
                            shape[i] = fill_value
                return torch.randn(
                    size=shape,
                    dtype=TORCH_DTYPES[x.dtype],
                    device=get_device(),
                )
            return x

        with device_scope("meta"):
            args_1, kwargs_1 = nest.map_structure(
                lambda x: convert_keras_tensor_to_torch(x, fill_value=83),
                (args, kwargs),
            )
            outputs_1 = fn(*args_1, **kwargs_1)

        outputs = outputs_1

        if none_in_shape:
            with device_scope("meta"):
                args_2, kwargs_2 = nest.map_structure(
                    lambda x: convert_keras_tensor_to_torch(x, fill_value=89),
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

        def convert_torch_to_keras_tensor(x):
            if is_tensor(x):
                return KerasTensor(x.shape, standardize_dtype(x.dtype))
            return x

        output_spec = nest.map_structure(convert_torch_to_keras_tensor, outputs)
    return output_spec


def cond(pred, true_fn, false_fn):
    if pred:
        return true_fn()
    return false_fn()


def vectorized_map(function, elements):
    return torch.vmap(function)(elements)


def scatter(indices, values, shape):
    indices = convert_to_tensor(indices)
    values = convert_to_tensor(values)
    zeros = torch.zeros(shape, dtype=values.dtype)

    index_length = indices.shape[-1]
    value_shape = shape[index_length:]
    indices = torch.reshape(indices, [-1, index_length])
    values = torch.reshape(values, [-1] + list(value_shape))

    for i in range(indices.shape[0]):
        index = indices[i]
        zeros[tuple(index)] += values[i]
    return zeros


def scatter_update(inputs, indices, updates):
    inputs = convert_to_tensor(inputs)
    indices = convert_to_tensor(indices, dtype="int64")
    updates = convert_to_tensor(updates)
    indices = torch.transpose(indices, 0, 1)

    inputs[tuple(indices)] = updates
    return inputs


def block_update(inputs, start_indices, updates):
    inputs = convert_to_tensor(inputs)
    start_indices = convert_to_tensor(start_indices, dtype="int64")
    updates = convert_to_tensor(updates)

    update_shape = updates.shape
    slices = [
        slice(start_index, start_index + update_length)
        for start_index, update_length in zip(start_indices, update_shape)
    ]
    inputs[slices] = updates
    return inputs


def while_loop(
    cond,
    body,
    loop_vars,
    maximum_iterations=None,
):
    current_iter = 0
    iteration_check = (
        lambda iter: maximum_iterations is None or iter < maximum_iterations
    )
    loop_vars = tuple([convert_to_tensor(v) for v in loop_vars])
    while cond(*loop_vars) and iteration_check(current_iter):
        loop_vars = body(*loop_vars)
        if not isinstance(loop_vars, (list, tuple)):
            loop_vars = (loop_vars,)
        loop_vars = tuple(loop_vars)
        current_iter += 1
    return loop_vars


def stop_gradient(variable):
    return variable.detach()
