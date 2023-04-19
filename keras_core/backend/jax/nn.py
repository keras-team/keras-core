from jax import nn as jnn
from jax import numpy as jnp

import flax
import jax


def relu(x):
    return jnn.relu(x)


def relu6(x):
    return jnn.relu6(x)


def sigmoid(x):
    return jnn.sigmoid(x)


def softplus(x):
    return jnn.softplus(x)


def softsign(x):
    return jnn.softsign(x)


def silu(x, beta=1.0):
    return jnn.silu(x, beta=beta)


def swish(x):
    return jnn.swish(x)


def log_sigmoid(x):
    return jnn.log_sigmoid(x)


def leaky_relu(x):
    return jnn.leaky_relu(x)


def hard_sigmoid(x):
    return jnn.hard_sigmoid(x)


def elu(x):
    return jnn.elu(x)


def selu(x):
    return jnn.selu(x)


def gelu(x):
    return jnn.gelu(x)


def softmax(x):
    return jnn.softmax(x)


def log_softmax(x):
    return jnn.log_softmax(x)


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


def depthwise_conv2d(x, filter, strides, padding):
    filter_shape = filter.shape
    feature_group_count = filter_shape[2]
    new_filter_shape = [
        filter_shape[0],
        filter_shape[1],
        1,
        filter_shape[2] * filter_shape[3],
    ]
    conv_filter = jnp.reshape(filter, new_filter_shape)
    return jnn.depthwise_conv2d(
        x,
        conv_filter,
        strides,
        padding,
        feature_group_count=feature_group_count,
    )


def separable_conv2d(
    x, depthwise_filter, pointwise_filter, strides, padding, dilations=None
):
    depthwise_conv_output = depthwise_conv2d(
        x,
        depthwise_filter,
        strides,
        padding,
        dilations=dilations,
    )
    return conv(depthwise_conv_output, pointwise_filter, [1, 1, 1, 1], padding)


def conv_transpose(x, filter, output_shape, padding, dilations=None):
    return jax.lax.conv_transpose(
        x,
        filter,
        output_shape,
        padding,
        rhs_dilation=dilations,
    )
