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
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "uint8": torch.uint8,
    "uint16": torch.int32,  # TODO: Torch doesn't have `uint16` dtype.
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
        return DEFAULT_DEVICE
    return device


def to_torch_dtype(dtype):
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
        ).to(get_device())

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
        value = convert_to_numpy(self.value)
        if dtype:
            return value.astype(dtype)
        return value

    @property
    def value(self):
        value = super().value
        # Create and use a symbolic tensor stub in symbolic calls.
        if get_device() == "meta" and str(value.device) != "meta":
            return torch.empty(
                size=value.shape,
                dtype=value.dtype,
                device="meta",
            )
        return value

    def __eq__(self, other):
        try:
            return super().__eq__(other)
        except Exception:
            return False


def convert_to_tensor(x, dtype=None):
    if is_tensor(x):
        if dtype is None:
            return x
        return x.to(to_torch_dtype(dtype))
    if isinstance(x, Variable):
        # TorchDynamo has bugs supporting nn.Parameter type check.
        # Return it directly instead of pass it to the rest of the logic in the
        # function.
        return x.value
    if isinstance(x, int):
        return torch.as_tensor(x, dtype=torch.int32, device=get_device())
    if isinstance(x, float):
        return torch.as_tensor(x, dtype=torch.float32, device=get_device())
    # Convert to np in case of any array-like that is not list or tuple.
    if not isinstance(x, (list, tuple)):
        x = np.array(x)
    elif len(x) > 0 and any(isinstance(x1, torch.Tensor) for x1 in x):
        # Handle list or tuple of torch tensors
        return torch.stack([convert_to_tensor(x1) for x1 in x])
    if isinstance(x, np.ndarray):
        if x.dtype == np.uint32:
            # Torch backend does not support uint32.
            x = x.astype(np.int64)
        dtype = dtype or x.dtype
    dtype = to_torch_dtype(dtype)
    return torch.as_tensor(x, dtype=dtype, device=get_device())


def convert_to_numpy(x):
    def transform(x):
        if is_tensor(x):
            if x.requires_grad:
                x = x.detach()
            # Tensor has to be moved to CPU before converting to numpy.
            if x.is_cuda:
                x = x.cpu()
        return np.array(x)

    if isinstance(x, (list, tuple)):
        return np.array([transform(e) for e in x])
    return transform(x)


def is_tensor(x):
    return torch.is_tensor(x)


def shape(x):
    return x.shape


def cast(x, dtype):
    dtype = to_torch_dtype(dtype)
    if isinstance(x, KerasVariable):
        x = x.value
    if is_tensor(x):
        if x.dtype == dtype:
            return x
        else:
            return x.to(dtype)
    return convert_to_tensor(x, dtype)


def name_scope(name):
    return contextlib.nullcontext()


# Shape / dtype inference util
def compute_output_spec(fn, *args, **kwargs):
    def has_none_shape(x):
        """Check for if a `KerasTensor` has dynamic shape."""
        if isinstance(x, KerasTensor):
            return None in x.shape
        return False

    def convert_keras_tensor_to_torch(x, fill_value=None):
        """Convert `KerasTensor`s to `torch.Tensor`s."""
        if isinstance(x, KerasTensor):
            shape = list(x.shape)
            if fill_value:
                for i, e in enumerate(shape):
                    if e is None:
                        shape[i] = fill_value
            return torch.ones(
                size=shape,
                dtype=TORCH_DTYPES[x.dtype],
                device=get_device(),
            )
        return x

    def convert_torch_to_keras_tensor(x):
        """Convert `torch.Tensor`s to `KerasTensor`s."""
        if is_tensor(x):
            return KerasTensor(x.shape, standardize_dtype(x.dtype))
        return x

    def symbolic_call(fn, args, kwargs, fill_value):
        """Call `fn` to infer output shape and dtype."""
        try:
            # First try instantiating all tensors on the `"meta"` device,
            # which  should give a "zero flop" way to trace shape, but does
            # not have universal support with torch operations.
            with device_scope("meta"):
                meta_args, meta_kwargs = nest.map_structure(
                    lambda x: convert_keras_tensor_to_torch(x, fill_value),
                    (args, kwargs),
                )
                return fn(*meta_args, **meta_kwargs)
        except:
            with device_scope(DEFAULT_DEVICE):
                # If the `"meta"` device placement fails, fall back to tracing
                # eagerly with tensors on the default device. This will be
                # more robust, but more expensive.
                eager_args, eager_kwargs = nest.map_structure(
                    lambda x: convert_keras_tensor_to_torch(x, fill_value),
                    (args, kwargs),
                )
                return fn(*eager_args, **eager_kwargs)

    with StatelessScope():
        outputs = symbolic_call(fn, args, kwargs, fill_value=83)

        none_in_shape = any(map(has_none_shape, nest.flatten((args, kwargs))))
        if none_in_shape:
            outputs_1 = outputs
            outputs_2 = symbolic_call(fn, args, kwargs, fill_value=89)

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

        output_spec = nest.map_structure(convert_torch_to_keras_tensor, outputs)
    return output_spec


def cond(pred, true_fn, false_fn):
    # When symbolic execution, take pred as true.
    if get_device() == "meta":
        return true_fn()

    if pred:
        return true_fn()
    return false_fn()


def vectorized_map(function, elements):
    return torch.vmap(function)(elements)


def scatter(indices, values, shape):
    indices = convert_to_tensor(indices)
    values = convert_to_tensor(values)
    zeros = torch.zeros(shape, dtype=values.dtype, device=get_device())

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


def slice(inputs, start_indices, shape):
    shape_dtype = to_torch_dtype("int64")
    inputs = convert_to_tensor(inputs)
    start_indices = convert_to_tensor(start_indices).to(shape_dtype)
    shape = convert_to_tensor(shape).to(shape_dtype)

    python_slice = __builtins__["slice"]
    slices = [
        python_slice(start_index, start_index + length)
        for start_index, length in zip(start_indices, shape)
    ]
    return inputs[slices]


def slice_update(inputs, start_indices, updates):
    shape_dtype = to_torch_dtype("int64")
    inputs = convert_to_tensor(inputs)
    start_indices = convert_to_tensor(start_indices).to(shape_dtype)
    updates = convert_to_tensor(updates)

    python_slice = __builtins__["slice"]
    slices = [
        python_slice(start_index, start_index + update_length)
        for start_index, update_length in zip(start_indices, updates.shape)
    ]
    outputs = torch.clone(inputs)
    outputs[slices] = updates
    return outputs


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
    # We can't use `.requires_grad_(False)` here since it only
    # works when the tensor is a leaf node in the graph.
    return variable.detach()
