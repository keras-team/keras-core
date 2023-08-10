import numpy as np
from keras_core import backend
from keras_core.utils.module_utils import tensorflow as tf


def validate_string_arg(
    input_data,
    allowable_strings,
    layer_name,
    arg_name,
    allow_none=False,
    allow_callables=False,
):
    """Validates the correctness of a string-based arg."""
    if allow_none and input_data is None:
        return
    elif allow_callables and callable(input_data):
        return
    elif isinstance(input_data, str) and input_data in allowable_strings:
        return
    else:
        allowed_args = "`None`, " if allow_none else ""
        allowed_args += "a `Callable`, " if allow_callables else ""
        allowed_args += f"or one of the following values: {allowable_strings}"
        if allow_callables:
            callable_note = (
                f"If restoring a model and `{arg_name}` is a custom callable, "
                "please ensure the callable is registered as a custom object. "
                "registering_the_custom_object for details. "
            )
        else:
            callable_note = ""
        raise ValueError(
            f"Unkown value for `{arg_name}` argument of layer {layer_name}. "
            f"{callable_note}Allowed values are: {allowed_args}. Received: "
            f"{input_data}"
        )


def ensure_tensor(inputs, dtype=None):
    """Ensures the input is a Tensor, SparseTensor or RaggedTensor."""
    if isinstance(inputs, (list, np.ndarray)):
        inputs = backend.convert_to_tensor(inputs, dtype)
    if dtype is not None and inputs.dtype != dtype:
        inputs = backend.cast(inputs, dtype)
    return inputs
