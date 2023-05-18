import numpy as np

from keras_core.backend.config import epsilon


def relu(x):
    return np.maximum(x, 0.0)


def relu6(x):
    return np.clip(x, 0.0, 6.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def softplus(x):
    return np.log(1.0 + np.exp(x))


def softsign(x):
    return x / (1.0 + np.abs(x))


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def swish(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def log_sigmoid(x):
    return np.log(1.0 / (1.0 + np.exp(-x)))


def leaky_relu(x, alpha=0.2):
    return np.maximum(x, alpha * x)


def hard_sigmoid(x):
    return np.maximum(0.0, np.minimum(1.0, x * 0.2 + 0.5))


def elu(x, alpha=1.0):
    return np.where(x >= 0.0, x, alpha * (np.exp(x) - 1.0))


def selu(x, alpha=1.67326, scale=1.0507):
    return scale * np.where(x >= 0.0, x, alpha * (np.exp(x) - 1.0))


def gelu(x):
    cdf = 0.5 * (
        1.0
        + np.tanh((np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3.0))))
    )
    return x * cdf


def softmax(x, axis=None):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x, axis=None):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return x - np.log(np.sum(exp_x, axis=axis, keepdims=True))


def generic_pool(inputs, pool_size, pool_fn, strides, padding, data_format):
    if padding != "valid":
        raise NotImplementedError(
            "Only 'valid' padding is currently supported."
        )
    if data_format != "channels_last":
        raise NotImplementedError(
            "Only 'channels_last' data format is currently supported."
        )
    rank = len(inputs.shape)
    if rank == 3:
        # 1D pooling
        N, L, C = inputs.shape
        pooled_L = (L - pool_size[0]) // strides[0] + 1
        pooled = np.empty((N, pooled_L, C))
        for i in range(pooled_L):
            start = i * strides[0]
            end = start + pool_size[0]
            pooled[:, i, :] = pool_fn(inputs[:, start:end, :], axis=1)

    elif rank == 4:
        # 2D pooling
        N, H, W, C = inputs.shape
        pooled_H = (H - pool_size[0]) // strides[0] + 1
        pooled_W = (W - pool_size[1]) // strides[1] + 1
        pooled = np.empty((N, pooled_H, pooled_W, C))
        for i in range(pooled_H):
            for j in range(pooled_W):
                start_H = i * strides[0]
                end_H = start_H + pool_size[0]
                start_W = j * strides[1]
                end_W = start_W + pool_size[1]
                pooled[:, i, j, :] = pool_fn(
                    inputs[:, start_H:end_H, start_W:end_W, :], axis=(1, 2)
                )

    elif rank == 5:
        # 3D pooling
        N, D, H, W, C = inputs.shape
        pooled_D = (D - pool_size[0]) // strides[0] + 1
        pooled_H = (H - pool_size[1]) // strides[1] + 1
        pooled_W = (W - pool_size[2]) // strides[2] + 1
        pooled = np.empty((N, pooled_D, pooled_H, pooled_W, C))
        for i in range(pooled_D):
            for j in range(pooled_H):
                for k in range(pooled_W):
                    start_D = i * strides[0]
                    end_D = start_D + pool_size[0]
                    start_H = j * strides[1]
                    end_H = start_H + pool_size[1]
                    start_W = k * strides[2]
                    end_W = start_W + pool_size[2]
                    pooled[:, i, j, k, :] = pool_fn(
                        inputs[
                            :, start_D:end_D, start_H:end_H, start_W:end_W, :
                        ],
                        axis=(1, 2, 3),
                    )

    else:
        raise NotImplementedError(
            "Pooling is not implemented for >3D tensors." f"Got {rank}D tensor."
        )

    return pooled


def max_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format="channels_last",
):
    num_spatial_dims = len(inputs.shape) - 2
    pool_size = (
        [pool_size] * num_spatial_dims
        if isinstance(pool_size, int)
        else pool_size
    )
    strides = pool_size if strides is None else strides
    strides = (
        [strides] * num_spatial_dims if isinstance(strides, int) else strides
    )
    return generic_pool(
        inputs, pool_size, np.max, strides, padding, data_format
    )


def average_pool(
    inputs,
    pool_size,
    strides,
    padding,
    data_format="channels_last",
):
    num_spatial_dims = len(inputs.shape) - 2
    pool_size = (
        [pool_size] * num_spatial_dims
        if isinstance(pool_size, int)
        else pool_size
    )
    strides = pool_size if strides is None else strides
    strides = (
        [strides] * num_spatial_dims if isinstance(strides, int) else strides
    )
    return generic_pool(
        inputs, pool_size, np.average, strides, padding, data_format
    )


# def conv(
#     inputs,
#     kernel,
#     strides=1,
#     padding="valid",
#     data_format="channel_last",
#     dilation_rate=1,
# ):
#     num_spatial_dims = inputs.ndim - 2
#     dimension_numbers = _convert_to_lax_conv_dimension_numbers(
#         num_spatial_dims,
#         data_format,
#         transpose=False,
#     )
#     strides = _convert_to_spatial_operand(
#         strides,
#         num_spatial_dims,
#         data_format,
#         include_batch_and_channels=False,
#     )
#     dilation_rate = _convert_to_spatial_operand(
#         dilation_rate,
#         num_spatial_dims,
#         data_format,
#         include_batch_and_channels=False,
#     )
#     return jax.lax.conv_general_dilated(
#         inputs,
#         kernel,
#         strides,
#         padding,
#         rhs_dilation=dilation_rate,
#         dimension_numbers=dimension_numbers,
#     )


# def depthwise_conv(
#     inputs,
#     kernel,
#     strides=1,
#     padding="valid",
#     data_format="channel_last",
#     dilation_rate=1,
# ):
#     num_spatial_dims = inputs.ndim - 2
#     dimension_numbers = _convert_to_lax_conv_dimension_numbers(
#         num_spatial_dims,
#         data_format,
#         transpose=False,
#     )
#     strides = _convert_to_spatial_operand(
#         strides,
#         num_spatial_dims,
#         data_format,
#         include_batch_and_channels=False,
#     )
#     dilation_rate = _convert_to_spatial_operand(
#         dilation_rate,
#         num_spatial_dims,
#         data_format,
#         include_batch_and_channels=False,
#     )
#     feature_group_count = (
#         inputs.shape[-1] if data_format == "channels_last" else inputs.shape[1]
#     )
#     kernel = jnp.reshape(
#         kernel,
#         kernel.shape[:-2] + (1, feature_group_count * kernel.shape[-1]),
#     )
#     return jax.lax.conv_general_dilated(
#         inputs,
#         kernel,
#         strides,
#         padding,
#         rhs_dilation=dilation_rate,
#         dimension_numbers=dimension_numbers,
#         feature_group_count=feature_group_count,
#     )


# def separable_conv(
#     inputs,
#     depthwise_kernel,
#     pointwise_kernel,
#     strides=1,
#     padding="valid",
#     data_format="channels_last",
#     dilation_rate=1,
# ):
#     depthwise_conv_output = depthwise_conv(
#         inputs,
#         depthwise_kernel,
#         strides,
#         padding,
#         data_format,
#         dilation_rate,
#     )
#     return conv(
#         depthwise_conv_output,
#         pointwise_kernel,
#         strides=1,
#         padding="valid",
#         data_format=data_format,
#         dilation_rate=dilation_rate,
#     )


# def conv_transpose(
#     inputs,
#     kernel,
#     strides=1,
#     padding="valid",
#     output_padding=None,
#     data_format="channels_last",
#     dilation_rate=1,
# ):
#     num_spatial_dims = inputs.ndim - 2
#     dimension_numbers = _convert_to_lax_conv_dimension_numbers(
#         num_spatial_dims,
#         data_format,
#         transpose=False,
#     )
#     strides = _convert_to_spatial_operand(
#         strides,
#         num_spatial_dims,
#         data_format,
#         include_batch_and_channels=False,
#     )
#     dilation_rate = _convert_to_spatial_operand(
#         dilation_rate,
#         num_spatial_dims,
#         data_format,
#         include_batch_and_channels=False,
#     )

#     if output_padding is not None:
#         raise ValueError(
#             "Custom `output_padding` is not supported yet, please set "
#             "`output_padding=None`."
#         )
#     padding = padding.upper()
#     return jax.lax.conv_transpose(
#         inputs,
#         kernel,
#         strides,
#         padding,
#         rhs_dilation=dilation_rate,
#         dimension_numbers=dimension_numbers,
#         transpose_kernel=True,
#    )


def one_hot(x, num_classes, axis=-1):
    one_hot_array = np.eye(num_classes)[x.reshape(-1)]
    output_shape = list(x.shape)
    output_shape.insert(axis, num_classes)
    return one_hot_array.reshape(output_shape)


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = np.array(target)
    output = np.array(output)

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
        log_prob = log_softmax(output, axis=axis)
    else:
        output = output / np.sum(output, axis, keepdims=True)
        output = np.clip(output, epsilon(), 1.0 - epsilon())
        log_prob = np.log(output)
    return -np.sum(target * log_prob, axis=axis)


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = np.array(target, dtype="int64")
    output = np.array(output)
    if len(target.shape) == len(output.shape) and target.shape[-1] == 1:
        target = np.squeeze(target, axis=-1)

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
        log_prob = log_softmax(output, axis=axis)
    else:
        output = output / np.sum(output, axis, keepdims=True)
        output = np.clip(output, epsilon(), 1.0 - epsilon())
        log_prob = np.log(output)
    target = one_hot(target, output.shape[axis], axis=axis)
    return -np.sum(target * log_prob, axis=axis)


def binary_crossentropy(target, output, from_logits=False):
    target = np.array(target)
    output = np.array(output)

    if target.shape != output.shape:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    if from_logits:
        output = sigmoid(output)

    output = np.clip(output, epsilon(), 1.0 - epsilon())
    bce = target * np.log(output)
    bce += (1.0 - target) * np.log(1.0 - output)
    return -bce
