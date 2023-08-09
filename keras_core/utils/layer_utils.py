import numpy as np

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


def listify_tensors(x):
    """Convert any tensors or numpy arrays to lists for config serialization."""
    if tf.is_tensor(x):
        x = x.numpy()
    if isinstance(x, np.ndarray):
        x = x.tolist()
    return x


def ensure_tensor(inputs, dtype=None):
    """Ensures the input is a Tensor, SparseTensor or RaggedTensor."""
    if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor, tf.SparseTensor)):
        inputs = tf.convert_to_tensor(inputs, dtype)
    if dtype is not None and inputs.dtype != dtype:
        inputs = tf.cast(inputs, dtype)
    return inputs


def is_ragged(tensor):
    """Returns true if `tensor` is a ragged tensor or ragged tensor value."""
    return isinstance(
        tensor, (tf.RaggedTensor, tf.compat.v1.ragged.RaggedTensorValue)
    )
