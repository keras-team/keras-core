import tensorflow as tf
from tensorflow import nn as tfnn


def relu(x):
    return tfnn.relu(x)


def relu6(x):
    return tfnn.relu6(x)


def sigmoid(x):
    return tfnn.sigmoid(x)


def softplus(x):
    return tf.math.softplus(x)


def softsign(x):
    return tfnn.softsign(x)


def silu(x, beta=1.0):
    return tfnn.silu(x, beta=beta)


def swish(x):
    return x * sigmoid(x)


def log_sigmoid(x):
    return tf.math.log_sigmoid(x)


def leaky_relu(x, negative_slope=0.2):
    return tfnn.leaky_relu(x, alpha=negative_slope)


def hard_sigmoid(x):
    x = x / 6.0 + 0.5
    return tf.clip_by_value(x, 0.0, 1.0)


def elu(x):
    return tfnn.elu(x)


def selu(x):
    return tfnn.selu(x)


def gelu(x, approximate=True):
    return tfnn.gelu(x, approximate)


def softmax(x, axis=None):
    return tfnn.softmax(x, axis=axis)


def log_softmax(x, axis=None):
    return tfnn.log_softmax(x, axis=axis)


def max_pool(x, window_shape, strides, padding):
    return tfnn.max_pool(x, window_shape, strides, padding)


def average_pool(x, window_shape, strides, padding):
    return tfnn.avg_pool(x, window_shape, strides, padding)


def conv(x, filter, strides, padding, data_format=None, dilations=None):
    return tfnn.convolution(
        x,
        filter,
        strides,
        padding,
        data_format=data_format,
        dilations=dilations,
    )


def depthwise_conv(
    x,
    filter,
    strides,
    padding,
    data_format=None,
    dilations=None,
):
    if len(x.shape) > 4:
        raise ValueError(
            "`depthwise_conv` does not support {len(x.shape)-2}D inputs yet."
        )
    if len(x.shape) == 3:
        # 1D depthwise conv.
        if data_format is None or data_format == "NWC":
            strides = (1,) + strides * 2 + (1,)
            spatial_start_dim = 1
            data_format = "NHWC"
        else:
            strides = (1, 1) + strides * 2
            spatial_start_dim = 2
            data_format = "NCHW"
        x = tf.expand_dims(x, spatial_start_dim)
        filter = tf.expand_dims(filter, axis=0)
        dilations = None if dilations is None else (1,) + dilations

        outputs = tf.nn.depthwise_conv2d(
            x,
            filter,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilations=dilations,
        )
        return tf.squeeze(outputs, [spatial_start_dim])

    return tfnn.depthwise_conv2d(
        x,
        filter,
        strides,
        padding,
        data_format=data_format,
        dilations=dilations,
    )


def separable_conv(
    x,
    depthwise_filter,
    pointwise_filter,
    strides,
    padding,
    data_format=None,
    dilations=None,
):
    if len(x.shape) > 4:
        raise ValueError(
            "`depthwise_conv` does not support {len(x.shape)-2}D inputs yet."
        )
    if len(x.shape) == 3:
        # 1D depthwise conv.
        if data_format is None or data_format == "NWC":
            strides = (1,) + strides * 2 + (1,)
            spatial_start_dim = 1
            data_format = "NHWC"
        else:
            strides = (1, 1) + strides * 2
            spatial_start_dim = 2
            data_format = "NCHW"
        x = tf.expand_dims(x, spatial_start_dim)
        depthwise_filter = tf.expand_dims(depthwise_filter, axis=0)
        pointwise_filter = tf.expand_dims(pointwise_filter, axis=0)
        dilations = None if dilations is None else (1,) + dilations

        outputs = tf.nn.separable_conv2d(
            x,
            depthwise_filter,
            pointwise_filter,
            strides,
            padding,
            data_format=data_format,
            dilations=dilations,
        )
        return tf.squeeze(outputs, [spatial_start_dim])

    return tfnn.separable_conv2d(
        x,
        depthwise_filter,
        pointwise_filter,
        strides,
        padding,
        data_format=data_format,
        dilations=dilations,
    )


def conv_transpose(
    x,
    filter,
    output_shape,
    strides,
    padding,
    data_format=None,
    dilations=None,
):
    return tfnn.conv_transpose(
        x,
        filter,
        output_shape,
        strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
    )
