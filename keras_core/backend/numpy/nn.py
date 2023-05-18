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
    if padding.lower() != "valid":
        raise NotImplementedError(
            "Only 'valid' padding is currently supported."
        )
    if data_format.lower() != "channels_last":
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


def conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format="channel_last",
    dilation_rate=1,
):
    if isinstance(strides, int):
        strides = (strides,) * len(inputs.shape[1:-1])
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * len(inputs.shape[1:-1])

    rank = len(inputs.shape)
    if padding.lower() != "valid":
        raise NotImplementedError(
            "Only 'VALID' padding is currently supported."
        )
    if data_format.lower() != "channels_last":
        raise NotImplementedError(
            "Only 'channels_last' data format is currently supported."
        )
    if any(rate != 1 for rate in dilation_rate):
        raise NotImplementedError(
            "Only dilation_rate of 1 is currently supported."
        )

    if rank == 3:
        # 1D convolution
        N, L, C = inputs.shape
        M, K = kernel.shape
        conv_L = (L - K) // strides[0] + 1
        convolved = np.empty((N, conv_L, M))
        for i in range(conv_L):
            start = i * strides[0]
            end = start + K
            convolved[:, i, :] = np.tensordot(
                inputs[:, start:end, :], kernel, axes=((1, 2), (1, 0))
            )

    elif rank == 4:
        # 2D convolution
        N, H, W, C = inputs.shape
        M, KH, KW = kernel.shape
        conv_H = (H - KH) // strides[0] + 1
        conv_W = (W - KW) // strides[1] + 1
        convolved = np.empty((N, conv_H, conv_W, M))
        for i in range(conv_H):
            for j in range(conv_W):
                start_H = i * strides[0]
                end_H = start_H + KH
                start_W = j * strides[1]
                end_W = start_W + KW
                convolved[:, i, j, :] = np.tensordot(
                    inputs[:, start_H:end_H, start_W:end_W, :],
                    kernel,
                    axes=((1, 2, 3), (1, 2, 0)),
                )

    elif rank == 5:
        # 3D convolution
        N, D, H, W, C = inputs.shape
        M, KD, KH, KW = kernel.shape
        conv_D = (D - KD) // strides[0] + 1
        conv_H = (H - KH) // strides[1] + 1
        conv_W = (W - KW) // strides[2] + 1
        convolved = np.empty((N, conv_D, conv_H, conv_W, M))
        for i in range(conv_D):
            for j in range(conv_H):
                for k in range(conv_W):
                    start_D = i * strides[0]
                    end_D = start_D + KD
                    start_H = j * strides[1]
                    end_H = start_H + KH
                    start_W = k * strides[2]
                    end_W = start_W + KW
                    convolved[:, i, j, k, :] = np.tensordot(
                        inputs[
                            :, start_D:end_D, start_H:end_H, start_W:end_W, :
                        ],
                        kernel,
                        axes=((1, 2, 3, 4), (1, 2, 3, 0)),
                    )

    else:
        raise NotImplementedError(
            "Convolution is not implemented for >3D tensors."
            f"Got {rank}D tensor."
        )

    return convolved


def depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format="channel_last",
    dilation_rate=1,
):
    if isinstance(strides, int):
        strides = (strides,) * len(inputs.shape[1:-1])
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * len(inputs.shape[1:-1])

    rank = len(inputs.shape)
    if padding.lower() != "valid":
        raise NotImplementedError(
            "Only 'VALID' padding is currently supported."
        )
    if data_format.lower() != "channels_last":
        raise NotImplementedError(
            "Only 'channels_last' data format is currently supported."
        )
    if any(rate != 1 for rate in dilation_rate):
        raise NotImplementedError(
            "Only dilation_rate of 1 is currently supported."
        )

    C = inputs.shape[-1]
    if C != kernel.shape[-1]:
        raise ValueError(
            "The number of input channels should be the same as the number of kernel channels."
        )

    if rank == 3:
        # 1D depthwise convolution
        N, L, _ = inputs.shape
        K, _ = kernel.shape
        conv_L = (L - K) // strides[0] + 1
        convolved = np.empty((N, conv_L, C))
        for c in range(C):
            for i in range(conv_L):
                start = i * strides[0]
                end = start + K
                convolved[:, i, c] = np.sum(
                    inputs[:, start:end, c] * kernel[:, c], axis=1
                )

    elif rank == 4:
        # 2D depthwise convolution
        N, H, W, _ = inputs.shape
        KH, KW, _ = kernel.shape
        conv_H = (H - KH) // strides[0] + 1
        conv_W = (W - KW) // strides[1] + 1
        convolved = np.empty((N, conv_H, conv_W, C))
        for c in range(C):
            for i in range(conv_H):
                for j in range(conv_W):
                    start_H = i * strides[0]
                    end_H = start_H + KH
                    start_W = j * strides[1]
                    end_W = start_W + KW
                    convolved[:, i, j, c] = np.sum(
                        inputs[:, start_H:end_H, start_W:end_W, c]
                        * kernel[:, :, c],
                        axis=(1, 2),
                    )

    elif rank == 5:
        # 3D depthwise convolution
        N, D, H, W, _ = inputs.shape
        KD, KH, KW, _ = kernel.shape
        conv_D = (D - KD) // strides[0] + 1
        conv_H = (H - KH) // strides[1] + 1
        conv_W = (W - KW) // strides[2] + 1
        convolved = np.empty((N, conv_D, conv_H, conv_W, C))
        for c in range(C):
            for i in range(conv_D):
                for j in range(conv_H):
                    for k in range(conv_W):
                        start_D = i * strides[0]
                        end_D = start_D + KD
                        start_H = j * strides[1]
                        end_H = start_H + KH
                        start_W = k * strides[2]
                        end_W = start_W + KW
                        convolved[:, i, j, k, c] = np.sum(
                            inputs[
                                :,
                                start_D:end_D,
                                start_H:end_H,
                                start_W:end_W,
                                c,
                            ]
                            * kernel[:, :, :, c],
                            axis=(1, 2, 3),
                        )

    else:
        raise NotImplementedError(
            "Depthwise convolution is not implemented for >3D tensors."
            f"Got {rank}D tensor."
        )

    return convolved


def separable_conv(
    inputs,
    depthwise_kernel,
    pointwise_kernel,
    strides=1,
    padding="valid",
    data_format="channel_last",
    dilation_rate=1,
):
    if isinstance(strides, int):
        strides = (strides,) * len(inputs.shape[1:-1])
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate,) * len(inputs.shape[1:-1])

    rank = len(inputs.shape)
    if padding.lower() != "valid":
        raise NotImplementedError(
            "Only 'VALID' padding is currently supported."
        )
    if data_format.lower() != "channels_last":
        raise NotImplementedError(
            "Only 'channels_last' data format is currently supported."
        )
    if any(rate != 1 for rate in dilation_rate):
        raise NotImplementedError(
            "Only dilation_rate of 1 is currently supported."
        )

    # Depthwise convolution
    depthwise_convolved = depthwise_conv(
        inputs, depthwise_kernel, strides, padding, data_format, dilation_rate
    )

    # Pointwise convolution
    N = depthwise_convolved.shape[0]
    M, C = pointwise_kernel.shape
    if C != depthwise_convolved.shape[-1]:
        raise ValueError(
            "The number of input channels should be the same as the number of kernel channels."
        )

    if rank == 3:
        # 1D pointwise convolution
        L, _ = depthwise_convolved.shape[1:]
        convolved = np.empty((N, L, M))
        for n in range(N):
            for i in range(L):
                convolved[n, i, :] = np.dot(
                    depthwise_convolved[n, i, :], pointwise_kernel.T
                )

    elif rank == 4:
        # 2D pointwise convolution
        H, W, _ = depthwise_convolved.shape[1:]
        convolved = np.empty((N, H, W, M))
        for n in range(N):
            for i in range(H):
                for j in range(W):
                    convolved[n, i, j, :] = np.dot(
                        depthwise_convolved[n, i, j, :], pointwise_kernel.T
                    )

    elif rank == 5:
        # 3D pointwise convolution
        D, H, W, _ = depthwise_convolved.shape[1:]
        convolved = np.empty((N, D, H, W, M))
        for n in range(N):
            for i in range(D):
                for j in range(H):
                    for k in range(W):
                        convolved[n, i, j, k, :] = np.dot(
                            depthwise_convolved[n, i, j, k, :],
                            pointwise_kernel.T,
                        )

    else:
        raise NotImplementedError(
            "Separable convolution is not implemented for >3D tensors."
            f"Got {rank}D tensor."
        )

    return convolved


def conv_transpose(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format="channel_last",
    output_padding=None,
):
    if isinstance(strides, int):
        strides = (strides,) * len(inputs.shape[1:-1])

    rank = len(inputs.shape)
    if padding.lower() != "valid":
        raise NotImplementedError(
            "Only 'VALID' padding is currently supported."
        )
    if data_format.lower() != "channels_last":
        raise NotImplementedError(
            "Only 'channels_last' data format is currently supported."
        )

    if output_padding is None:
        output_padding = (0,) * len(inputs.shape[1:-1])

    M, C = kernel.shape[:2]
    if C != inputs.shape[-1]:
        raise ValueError(
            "The number of input channels should be the same as the number of kernel channels."
        )

    if rank == 3:
        # 1D convolution transpose
        N, L, _ = inputs.shape
        K, _ = kernel.shape[2:]
        conv_L = (L - 1) * strides[0] + K + output_padding[0]
        convolved = np.zeros((N, conv_L, M))
        for i in range(L):
            start = i * strides[0]
            end = start + K
            convolved[:, start:end, :] += np.tensordot(
                inputs[:, i, :], kernel, axes=(1, 1)
            )

    elif rank == 4:
        # 2D convolution transpose
        N, H, W, _ = inputs.shape
        KH, KW, _ = kernel.shape[2:]
        conv_H = (H - 1) * strides[0] + KH + output_padding[0]
        conv_W = (W - 1) * strides[1] + KW + output_padding[1]
        convolved = np.zeros((N, conv_H, conv_W, M))
        for i in range(H):
            for j in range(W):
                start_H = i * strides[0]
                end_H = start_H + KH
                start_W = j * strides[1]
                end_W = start_W + KW
                convolved[:, start_H:end_H, start_W:end_W, :] += np.tensordot(
                    inputs[:, i, j, :], kernel, axes=(1, 1)
                )

    elif rank == 5:
        # 3D convolution transpose
        N, D, H, W, _ = inputs.shape
        KD, KH, KW, _ = kernel.shape[2:]
        conv_D = (D - 1) * strides[0] + KD + output_padding[0]
        conv_H = (H - 1) * strides[1] + KH + output_padding[1]
        conv_W = (W - 1) * strides[2] + KW + output_padding[2]
        convolved = np.zeros((N, conv_D, conv_H, conv_W, M))
        for i in range(D):
            for j in range(H):
                for k in range(W):
                    start_D = i * strides[0]
                    end_D = start_D + KD
                    start_H = j * strides[1]
                    end_H = start_H + KH
                    start_W = k * strides[2]
                    end_W = start_W + KW
                    convolved[
                        :, start_D:end_D, start_H:end_H, start_W:end_W, :
                    ] += np.tensordot(
                        inputs[:, i, j, k, :], kernel, axes=(1, 1)
                    )

    else:
        raise NotImplementedError(
            "Convolution transpose is not implemented for >3D tensors."
            f"Got {rank}D tensor."
        )

    return convolved


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
