from tensorflow import nn as tfnn
import tensorflow as tf


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
    return tfnn.swish(x)


def log_sigmoid(x):
    return tfnn.log_sigmoid(x)


def leaky_relu(x):
    return tfnn.leaky_relu(x)


def hard_sigmoid(x):
    return tfnn.hard_sigmoid(x)


def elu(x):
    return tfnn.elu(x)


def selu(x):
    return tfnn.selu(x)


def gelu(x):
    return tfnn.gelu(x)


def softmax(x):
    return tfnn.softmax(x)


def log_softmax(x):
    return tfnn.log_softmax(x)


def max_pool(x, window_shape, strides, padding):
    return tfnn.max_pool(x, window_shape, strides, padding)


def average_pool(x, window_shape, strides, padding):
    return tfnn.avg_pool(x, window_shape, strides, padding)


def convert_data_format(data_format, ndim):
    if data_format == "channels_last":
        if ndim == 3:
            return "NWC"
        elif ndim == 4:
            return "NHWC"
        elif ndim == 5:
            return "NDHWC"
        else:
            raise ValueError(
                f"Input rank not supported: {ndim}. "
                "Expected values are [3, 4, 5]"
            )
    elif data_format == "channels_first":
        if ndim == 3:
            return "NCW"
        elif ndim == 4:
            return "NCHW"
        elif ndim == 5:
            return "NCDHW"
        else:
            raise ValueError(
                f"Input rank not supported: {ndim}. "
                "Expected values are [3, 4, 5]"
            )
    else:
        raise ValueError(
            f"Invalid data_format: {data_format}. "
            'Expected values are ["channels_first", "channels_last"]'
        )


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
        depthwise_filter = tf.expand_dims(depthwise_filter, axis=0)
        dilations = None if dilations is None else (1,) + dilations

        outputs = tf.nn.depthwise_conv2d(
            x,
            depthwise_filter,
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
