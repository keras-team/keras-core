import torch
import torch.nn.functional as F

from keras_core.backend.config import epsilon


def relu(x):
    return F.relu(x)


def relu6(x):
    return F.relu6(x)


def sigmoid(x):
    return F.sigmoid(x)


def tanh(x):
    return F.tanh(x)


def softplus(x):
    return F.softplus(x)


def softsign(x):
    return F.soft_sign(x)


def silu(x):
    return F.silu(x)


def swish(x):
    return x * F.sigmoid(x)


def log_sigmoid(x):
    return F.logsigmoid(x)


def leaky_relu(x, negative_slope=0.2):
    return F.leaky_relu(x, negative_slope=negative_slope)


def hard_sigmoid(x):
    return F.hardsigmoid(x)


def elu(x):
    return F.elu(x)


def selu(x):
    return F.selu(x)


def gelu(x, approximate=True):
    return F.gelu(x, approximate)


def softmax(x, axis=None):
    return F.softmax(x, dim=axis)


def log_softmax(x, axis=-1):
    return F.log_softmax(x, dim=axis)


def max_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format="channels_last",
):
    raise NotImplementedError("`max_pool` not yet implemeneted for PyTorch Backend")


def average_pool(
    inputs,
    pool_size,
    strides,
    padding,
    data_format="channels_last",
):
    raise NotImplementedError("`average_pool` not yet implemeneted for PyTorch Backend")


def conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format="channels_last",
    dilation_rate=1,
):
    raise NotImplementedError("`conv` not yet implemeneted for PyTorch Backend")


def depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format="channels_last",
    dilation_rate=1,
):
    raise NotImplementedError("`depthwise_conv` not yet implemeneted for PyTorch Backend")


def separable_conv(
    inputs,
    depthwise_kernel,
    pointwise_kernel,
    strides=1,
    padding="valid",
    data_format="channels_last",
    dilation_rate=1,
):
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
    data_format="channels_last",
    dilation_rate=1,
):
    raise NotImplementedError("`conv_transpose` not yet implemeneted for PyTorch backend")


def one_hot(x, num_classes, axis=-1):
    if axis != -1 or axis != x.shape[-1]:
        raise ValueError("`one_hot` is only implemented for last axis for PyTorch backend")
    return F.one_hot(x, num_classes)


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = torch.as_tensor(target)
    output = torch.as_tensor(output)

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
        log_prob = F.log_softmax(output, dim=axis)
    else:
        output = output / torch.sum(output, dim=axis, keepdim=True)
        output = torch.clip(output, epsilon(), 1.0 - epsilon())
        log_prob = torch.log(output)
    return -torch.sum(target * log_prob, dim=axis)


def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = torch.as_tensor(target, dtype=torch.long)
    output = torch.as_tensor(output)

    if len(target.shape) == len(output.shape) and target.shape[-1] == 1:
        target = torch.squeeze(target, dim=-1)

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
        log_prob = F.log_softmax(output, dim=axis)
    else:
        output = output / torch.sum(output, dim=axis, keepdim=True)
        output = torch.clip(output, epsilon(), 1.0 - epsilon())
        log_prob = torch.log(output)
    target = one_hot(target, output.shape[axis], axis=axis)
    return -torch.sum(target * log_prob, dim=axis)


def binary_crossentropy(target, output, from_logits=False):
    # TODO: `torch.as_tensor` has device arg. Need to think how to pass it.
    target = torch.as_tensor(target)
    output = torch.as_tensor(output)

    if target.shape != output.shape:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    if from_logits:
        output = F.sigmoid(output)

    output = torch.clip(output, epsilon(), 1.0 - epsilon())
    bce = target * torch.log(output)
    bce += (1.0 - target) * torch.log(1.0 - output)
    return -bce
