import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax import nn as jnn

from keras_core.backend import standardize_data_format
from keras_core.backend.common.backend_utils import (
    compute_conv_transpose_padding,
)
from keras_core.backend.config import epsilon
from keras_core.backend.jax.core import convert_to_tensor


def relu(x):
    return jnn.relu(x)


def relu6(x):
    return jnn.relu6(x)


def sigmoid(x):
    return jnn.sigmoid(x)


def tanh(x):
    return jnn.tanh(x)


def softplus(x):
    return jnn.softplus(x)


def softsign(x):
    return jnn.soft_sign(x)


def silu(x):
    return jnn.silu(x)


def swish(x):
    return jnn.swish(x)


def log_sigmoid(x):
    return jnn.log_sigmoid(x)


def leaky_relu(x, negative_slope=0.2):
    return jnn.leaky_relu(x, negative_slope=negative_slope)


def hard_sigmoid(x):
    return jnn.hard_sigmoid(x)


def elu(x, alpha=1.0):
    return jnn.elu(x, alpha=alpha)


def selu(x):
    return jnn.selu(x)


def gelu(x, approximate=True):
    return jnn.gelu(x, approximate)


def softmax(x, axis=None):
    return jnn.softmax(x, axis=axis)


def log_softmax(x, axis=-1):
    return jnn.log_softmax(x, axis=axis)


def _convert_to_spatial_operand(
    x,
    num_spatial_dims,
    data_format="channels_last",
    include_batch_and_channels=True,
):
    # Helper function that converts an operand to a spatial operand.
    x = (x,) * num_spatial_dims if isinstance(x, int) else x
    if not include_batch_and_channels:
        return x
    if data_format == "channels_last":
        x = (1,) + x + (1,)
    else:
        x = (1,) + (1,) + x
    return x


def _pool(
    inputs,
    initial_value,
    reduce_fn,
    pool_size,
    strides=None,
    padding="valid",
):
    """Helper function to define pooling functions.

    Args:
        inputs: input data of shape `N+2`.
        initial_value: the initial value for the reduction.
        reduce_fn: a reduce function of the form `(T, T) -> T`.
        pool_size: a sequence of `N` integers, representing the window size to
            reduce over.
        strides: a sequence of `N` integers, representing the inter-window
            strides (default: `(1, ..., 1)`).
        padding: either the string `same` or `valid`.

    Returns:
        The output of the reduction for each window slice.
    """
    if padding not in ("same", "valid"):
        raise ValueError(
            f"Invalid padding '{padding}', must be 'same' or 'valid'."
        )
    padding = padding.upper()
    return lax.reduce_window(
        inputs,
        initial_value,
        reduce_fn,
        pool_size,
        strides,
        padding,
    )


def max_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    data_format = standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2
    pool_size = _convert_to_spatial_operand(
        pool_size, num_spatial_dims, data_format
    )
    strides = pool_size if strides is None else strides
    strides = _convert_to_spatial_operand(
        strides, num_spatial_dims, data_format
    )
    return _pool(inputs, -jnp.inf, lax.max, pool_size, strides, padding)


def average_pool(
    inputs,
    pool_size,
    strides,
    padding,
    data_format=None,
):
    data_format = standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2
    pool_size = _convert_to_spatial_operand(
        pool_size, num_spatial_dims, data_format
    )
    strides = pool_size if strides is None else strides
    strides = _convert_to_spatial_operand(
        strides, num_spatial_dims, data_format
    )

    pooled = _pool(inputs, 0.0, lax.add, pool_size, strides, padding)
    if padding == "valid":
        # Avoid the extra reduce_window.
        return pooled / np.prod(pool_size)
    else:
        # Count the number of valid entries at each input point, then use that
        # for computing average. Assumes that any two arrays of same shape will
        # be padded the same. Avoid broadcasting on axis where pooling is
        # skipped.
        shape = [
            (a if b != 1 else 1) for (a, b) in zip(inputs.shape, pool_size)
        ]
        window_counts = _pool(
            jnp.ones(shape, inputs.dtype),
            0.0,
            lax.add,
            pool_size,
            strides,
            padding,
        )
        return pooled / window_counts


def _convert_to_lax_conv_dimension_numbers(
    num_spatial_dims,
    data_format="channels_last",
    transpose=False,
):
    """Create a `lax.ConvDimensionNumbers` for the given inputs."""
    num_dims = num_spatial_dims + 2

    if data_format == "channels_last":
        spatial_dims = tuple(range(1, num_dims - 1))
        inputs_dn = (0, num_dims - 1) + spatial_dims
    else:
        spatial_dims = tuple(range(2, num_dims))
        inputs_dn = (0, 1) + spatial_dims

    if transpose:
        kernel_dn = (num_dims - 2, num_dims - 1) + tuple(range(num_dims - 2))
    else:
        kernel_dn = (num_dims - 1, num_dims - 2) + tuple(range(num_dims - 2))

    return lax.ConvDimensionNumbers(
        lhs_spec=inputs_dn, rhs_spec=kernel_dn, out_spec=inputs_dn
    )


def conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    data_format = standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2
    dimension_numbers = _convert_to_lax_conv_dimension_numbers(
        num_spatial_dims,
        data_format,
        transpose=False,
    )
    strides = _convert_to_spatial_operand(
        strides,
        num_spatial_dims,
        data_format,
        include_batch_and_channels=False,
    )
    dilation_rate = _convert_to_spatial_operand(
        dilation_rate,
        num_spatial_dims,
        data_format,
        include_batch_and_channels=False,
    )
    if data_format == "channels_last":
        channels = inputs.shape[-1]
    else:
        channels = inputs.shape[1]
    kernel_in_channels = kernel.shape[-2]
    if channels % kernel_in_channels > 0:
        raise ValueError(
            "The number of input channels must be evenly divisible by "
            f"kernel's in_channels. Received input channels {channels} and "
            f"kernel in_channels {kernel_in_channels}. "
        )
    feature_group_count = channels // kernel_in_channels
    return jax.lax.conv_general_dilated(
        convert_to_tensor(inputs),
        convert_to_tensor(kernel),
        strides,
        padding,
        rhs_dilation=dilation_rate,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
    )


def depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    data_format = standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2
    dimension_numbers = _convert_to_lax_conv_dimension_numbers(
        num_spatial_dims,
        data_format,
        transpose=False,
    )
    strides = _convert_to_spatial_operand(
        strides,
        num_spatial_dims,
        data_format,
        include_batch_and_channels=False,
    )
    dilation_rate = _convert_to_spatial_operand(
        dilation_rate,
        num_spatial_dims,
        data_format,
        include_batch_and_channels=False,
    )
    feature_group_count = (
        inputs.shape[-1] if data_format == "channels_last" else inputs.shape[1]
    )
    kernel = jnp.reshape(
        kernel,
        kernel.shape[:-2] + (1, feature_group_count * kernel.shape[-1]),
    )
    return jax.lax.conv_general_dilated(
        inputs,
        kernel,
        strides,
        padding,
        rhs_dilation=dilation_rate,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
    )


def separable_conv(
    inputs,
    depthwise_kernel,
    pointwise_kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    data_format = standardize_data_format(data_format)
    depthwise_conv_output = depthwise_conv(
        inputs,
        depthwise_kernel,
        strides,
        padding,
        data_format,
        dilation_rate,
    )
    return conv(
        depthwise_conv_output,
        pointwise_kernel,
        strides=1,
        padding="valid",
        data_format=data_format,
        dilation_rate=dilation_rate,
    )


def conv_transpose(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    output_padding=None,
    data_format=None,
    dilation_rate=1,
):
    data_format = standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2
    padding_values = compute_conv_transpose_padding(
        inputs.shape,
        kernel.shape,
        strides,
        padding,
        output_padding,
        data_format,
        dilation_rate,
    )
    dimension_numbers = _convert_to_lax_conv_dimension_numbers(
        num_spatial_dims,
        data_format,
        transpose=False,
    )
    strides = _convert_to_spatial_operand(
        strides,
        num_spatial_dims,
        data_format,
        include_batch_and_channels=False,
    )
    dilation_rate = _convert_to_spatial_operand(
        dilation_rate,
        num_spatial_dims,
        data_format,
        include_batch_and_channels=False,
    )

    return jax.lax.conv_transpose(
        inputs,
        kernel,
        strides,
        padding=padding_values,
        rhs_dilation=dilation_rate,
        dimension_numbers=dimension_numbers,
        transpose_kernel=True,
    )


def one_hot(x, num_classes, axis=-1):
    return jnn.one_hot(x, num_classes, axis=axis)


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = jnp.array(target)
    output = jnp.array(output)

    if target.shape != output.shape:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )
    if len(target.shape) < 1:
        raise ValueError(
            "Arguments `target` and `output` must be at least rank 1. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    if from_logits:
        log_prob = jax.nn.log_softmax(output, axis=axis)
    else:
        output = output / jnp.sum(output, axis, keepdims=True)
        output = jnp.clip(output, epsilon(), 1.0 - epsilon())
        log_prob = jnp.log(output)
    return -jnp.sum(target * log_prob, axis=axis)


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = jnp.array(target, dtype="int32")
    output = jnp.array(output)
    if len(target.shape) == len(output.shape) and target.shape[-1] == 1:
        target = jnp.squeeze(target, axis=-1)

    if len(output.shape) < 1:
        raise ValueError(
            "Argument `output` must be at least rank 1. "
            "Received: "
            f"output.shape={output.shape}"
        )
    if target.shape != output.shape[:-1]:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape "
            "up until the last dimension: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )
    if from_logits:
        log_prob = jax.nn.log_softmax(output, axis=axis)
    else:
        output = output / jnp.sum(output, axis, keepdims=True)
        output = jnp.clip(output, epsilon(), 1.0 - epsilon())
        log_prob = jnp.log(output)
    target = jnn.one_hot(target, output.shape[axis], axis=axis)
    return -jnp.sum(target * log_prob, axis=axis)


def binary_crossentropy(target, output, from_logits=False):
    target = jnp.array(target)
    output = jnp.array(output)

    if target.shape != output.shape:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    if from_logits:
        log_logits = jax.nn.log_sigmoid(output)
        log_neg_logits = jax.nn.log_sigmoid(-output)
        return -1.0 * target * log_logits - (1.0 - target) * log_neg_logits

    output = jnp.clip(output, epsilon(), 1.0 - epsilon())
    bce = target * jnp.log(output)
    bce += (1.0 - target) * jnp.log(1.0 - output)
    return -bce
