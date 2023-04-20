import jax
from jax import nn as jnn
from jax import numpy as jnp


def relu(x):
    return jnn.relu(x)


def relu6(x):
    return jnn.relu6(x)


def sigmoid(x):
    return jnn.sigmoid(x)


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


def elu(x):
    return jnn.elu(x)


def selu(x):
    return jnn.selu(x)


def gelu(x, approximate=True):
    return jnn.gelu(x, approximate)


def softmax(x):
    return jnn.softmax(x)


def log_softmax(x, axis=-1):
    return jnn.log_softmax(x, axis=axis)


def max_pool(x, window_shape, strides, padding):
    # TODO: Implement `max_pool` with JAX ops.
    raise NotImplementedError


def average_pool(x, window_shape, strides, padding):
    # TODO: Implement `average_pool` with JAX ops.
    raise NotImplementedError


def conv(x, filter, strides, padding, dilations=None):
    return jax.lax.conv_general_dilated(
        x, filter, strides, padding, rhs_dilation=dilations
    )


def depthwise_conv(x, filter, strides, padding):
    # TODO: Implement `depthwise_conv` with `conv_general_dilated`.
    raise NotImplementedError


def separable_conv(x, depthwise_filter, pointwise_filter, strides, padding):
    # TODO: Implement `separable_conv` with `conv_general_dilated`.
    raise NotImplementedError


def conv_transpose(x, filter, output_shape, padding, dilations=None):
    return jax.lax.conv_transpose(
        x,
        filter,
        output_shape,
        padding,
        rhs_dilation=dilations,
    )
