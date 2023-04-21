"""
relu
relu6
sigmoid
softplus
softsign
silu
swish
log_sigmoid
leaky_relu
hard_sigmoid
elu
selu
gelu
softmax
log_softmax

max_pooling
average_pooling
conv
depthwise_conv
separable_conv
conv_transpose

ctc ??
"""

from keras_core import backend
from keras_core.backend import KerasTensor
from keras_core.backend import any_symbolic_tensors
from keras_core.operations.operation import Operation

import math
import numpy as np


class Relu(Operation):
    def call(self, x):
        return backend.nn.relu(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, x.dtype)


def relu(x):
    if any_symbolic_tensors((x,)):
        return Relu().symbolic_call(x)
    return backend.nn.relu(x)


class Relu6(Operation):
    def call(self, x):
        return backend.nn.relu6(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape, x.dtype)


def relu6(x):
    if any_symbolic_tensors((x,)):
        return Relu6().symbolic_call(x)
    return backend.nn.relu6(x)


class Sigmoid(Operation):
    def call(self, x):
        return backend.nn.sigmoid(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def sigmoid(x):
    if any_symbolic_tensors((x,)):
        return Sigmoid().symbolic_call(x)
    return backend.nn.sigmoid(x)


class Softplus(Operation):
    def call(self, x):
        return backend.nn.softplus(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def softplus(x):
    if any_symbolic_tensors((x,)):
        return Softplus().symbolic_call(x)
    return backend.nn.softplus(x)


class Softsign(Operation):
    def call(self, x):
        return backend.nn.softsign(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def softsign(x):
    if any_symbolic_tensors((x,)):
        return Softsign().symbolic_call(x)
    return backend.nn.softsign(x)


class Silu(Operation):
    def call(self, x):
        return backend.nn.silu(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def silu(x):
    if any_symbolic_tensors((x,)):
        return Silu().symbolic_call(x)
    return backend.nn.silu(x)


class Swish(Operation):
    def call(self, x):
        return backend.nn.swish(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def swish(x):
    if any_symbolic_tensors((x,)):
        return Swish().symbolic_call(x)
    return backend.nn.swish(x)


class LogSigmoid(Operation):
    def call(self, x):
        return backend.nn.log_sigmoid(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def log_sigmoid(x):
    if any_symbolic_tensors((x,)):
        return LogSigmoid().symbolic_call(x)
    return backend.nn.log_sigmoid(x)


class LeakyRelu(Operation):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope

    def call(self, x):
        return backend.nn.leaky_relu(x, self.negative_slope)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def leaky_relu(x, negative_slope=0.2):
    if any_symbolic_tensors((x,)):
        return LeakyRelu(negative_slope).symbolic_call(x)
    return backend.nn.leaky_relu(x, negative_slope=negative_slope)


class HardSigmoid(Operation):
    def call(self, x):
        return backend.nn.hard_sigmoid(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def hard_sigmoid(x):
    if any_symbolic_tensors((x,)):
        return HardSigmoid().symbolic_call(x)
    return backend.nn.hard_sigmoid(x)


class Elu(Operation):
    def call(self, x):
        return backend.nn.elu(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def elu(x):
    if any_symbolic_tensors((x,)):
        return Elu().symbolic_call(x)
    return backend.nn.elu(x)


class Selu(Operation):
    def call(self, x):
        return backend.nn.selu(x)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def selu(x):
    if any_symbolic_tensors((x,)):
        return Selu().symbolic_call(x)
    return backend.nn.selu(x)


class Gelu(Operation):
    def __init__(self, approximate=True):
        super().__init__()
        self.approximate = approximate

    def call(self, x):
        return backend.nn.gelu(x, self.approximate)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def gelu(x, approximate=True):
    if any_symbolic_tensors((x,)):
        return Gelu(approximate).symbolic_call(x)
    return backend.nn.gelu(x, approximate)


class Softmax(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis if axis is not None else -1

    def call(self, x):
        return backend.nn.softmax(x, axis=self.axis)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def softmax(x, axis=None):
    if any_symbolic_tensors((x,)):
        return Softmax(axis).symbolic_call(x)
    return backend.nn.softmax(x)


class LogSoftmax(Operation):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis if axis is not None else -1

    def call(self, x):
        return backend.nn.log_softmax(x, axis=self.axis)

    def compute_output_spec(self, x):
        return KerasTensor(x.shape)


def log_softmax(x, axis=None):
    if any_symbolic_tensors((x,)):
        return LogSoftmax(axis).symbolic_call(x)
    return backend.nn.log_softmax(x, axis=axis)


class MaxPool(Operation):
    def __init__(
        self,
        pool_size,
        strides,
        padding="valid",
        data_format="channels_last",
    ):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format

    def call(self, inputs):
        return backend.nn.max_pool(
            inputs,
            self.pool_size,
            self.strides,
            self.padding,
            self.data_format,
        )

    def compute_output_spec(self, inputs):
        input_shape = np.array(inputs.shape)
        if self.data_format == "channels_last":
            spacial_shape = input_shape[1:-1]
        else:
            spacial_shape = input_shape[2:]
        pool_size = np.array(self.pool_size)
        if self.padding == "valid":
            output_spacial_shape = (
                np.floor((spacial_shape - self.pool_size) / self.strides) + 1
            )
            negative_in_shape = np.all(output_spacial_shape < 0)
            if negative_in_shape:
                raise ValueError(
                    "Computed output size would be negative. Received "
                    f"`inputs.shape={input_shape}` and `pool_size={pool_size}`."
                )
        elif self.padding == "same":
            output_spacial_shape = (
                np.floor((spacial_shape - 1) / self.strides) + 1
            )

        if self.data_format == "channels_last":
            output_shape = (
                input_shape[0] + output_spacial_shape + input_shape[-1]
            )
        else:
            output_shape = input_shape[:2] + output_spacial_shape
        return KerasTensor(output_shape, dtype=inputs.dtype)


def max_pool(
    x,
    pool_size,
    strides,
    padding="valid",
    data_format="channels_last",
):
    if any_symbolic_tensors((x,)):
        return MaxPool(
            pool_size,
            strides,
            padding,
            data_format,
        ).symbolic_call(x)
    return backend.nn.max_pool(x, pool_size, strides, padding, data_format)


class AveragePool(Operation):
    def __init__(
        self,
        pool_size,
        strides,
        padding="valid",
        data_format="channels_last",
    ):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format

    def call(self, inputs):
        return backend.nn.average_pool(
            inputs,
            self.pool_size,
            self.strides,
            self.padding,
            self.data_format,
        )

    def compute_output_spec(self, inputs):
        input_shape = np.array(inputs.shape)
        if self.data_format == "channels_last":
            spacial_shape = input_shape[1:-1]
        else:
            spacial_shape = input_shape[2:]
        pool_size = np.array(self.pool_size)
        if self.padding == "valid":
            output_spacial_shape = (
                np.floor((spacial_shape - self.pool_size) / self.strides) + 1
            )
            negative_in_shape = np.all(output_spacial_shape < 0)
            if negative_in_shape:
                raise ValueError(
                    "Computed output size would be negative. Received "
                    f"`inputs.shape={input_shape}` and `pool_size={pool_size}`."
                )
        elif self.padding == "same":
            output_spacial_shape = (
                np.floor((spacial_shape - 1) / self.strides) + 1
            )

        if self.data_format == "channels_last":
            output_shape = (
                [input_shape[0]] + output_spacial_shape + [input_shape[-1]]
            )
        else:
            output_shape = input_shape[:2] + output_spacial_shape
        return KerasTensor(output_shape, dtype=inputs.dtype)


def average_pool(
    x,
    pool_size,
    strides,
    padding="valid",
    data_format="channels_last",
):
    if any_symbolic_tensors((x,)):
        return AveragePool(
            pool_size,
            strides,
            padding,
            data_format,
        ).symbolic_call(x)
    return backend.nn.average_pool(x, pool_size, strides, padding, data_format)


class Conv(Operation):
    def __init__(
        self,
        kernel,
        strides,
        padding="valid",
        data_format="channel_last",
        dilations=None,
    ):
        super().__init__()
        self.kernel = kernel
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilations = dilations

    def call(self, inputs):
        return backend.nn.conv(
            inputs,
            self.kernel,
            self.strides,
            self.padding,
            self.data_format,
            self.dilations,
        )

    def compute_output_spec(self, inputs):
        input_shape = inputs.shape
        if self.data_format == "channels_last":
            spacial_shape = input_shape[1:-1]
        else:
            spacial_shape = input_shape[2:]
        if len(self.kernel.shape) != len(input_shape):
            raise ValueError(
                "Kernel shape must have the same length as input, but received "
                f"kernel of shape {self.kernel.shape} and "
                f"input of shape {input_shape}."
            )
        if (
            self.dilations is not None
            and len(self.dilations) != 1
            and len(self.dilations) != len(spacial_shape)
        ):
            raise ValueError(
                "Dilation must be None, scalar or tuple/list of length of "
                "inputs' spacial shape, but received "
                f"`dilations={self.dilations}` and input of shape {input_shape}."
            )
        spacial_shape = np.array(spacial_shape)
        kernel_spacial_shape = np.array(self.kernel.shape[:2])
        dilations = np.array(self.dilations)
        if self.padding == "valid":
            output_spacial_shape = (
                np.floor(
                    (spacial_shape - dilations * (kernel_spacial_shape - 1) - 1)
                    / self.strides
                )
                + 1
            )
            negative_in_shape = np.all(output_spacial_shape < 0)
            if negative_in_shape:
                raise ValueError(
                    "Computed output size would be negative. Received "
                    f"`inputs shape={inputs.shape}`, "
                    f"`kernel spacial size={self.kernel.size}`, "
                    f"`dilations={self.dilations}`."
                )
        elif self.padding == "same":
            output_spacial_shape = (
                np.floor((spacial_shape - 1) / self.strides) + 1
            )

        if self.data_format == "channels_last":
            output_shape = (
                [input_shape[0]]
                + output_spacial_shape
                + [self.kernel.shape[-1]]
            )
        else:
            output_shape = [
                input_shape[0],
                self.kernel.shape[-1],
            ] + output_spacial_shape
        return KerasTensor(output_shape, dtype=inputs.dtype)


def conv(
    x,
    kernel,
    strides,
    padding="valid",
    data_format="channels_last",
    dilations=None,
):
    if any_symbolic_tensors((x,)):
        return Conv(
            kernel, strides, padding, data_format, dilations
        ).symbolic_call(x)
    return backend.nn.conv(x, kernel, strides, padding, data_format, dilations)


class DepthwiseConv(Operation):
    def __init__(
        self,
        kernel,
        strides,
        padding="valid",
        data_format="channels_last",
        dilations=None,
    ):
        super().__init__()
        self.kernel = kernel
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilations = dilations

    def call(self, inputs):
        return backend.nn.depthwise_conv(
            inputs,
            self.kernel,
            self.strides,
            self.padding,
            self.data_format,
            self.dilations,
        )

    def compute_output_spec(self, inputs):
        input_shape = inputs.shape
        if self.data_format == "channels_last":
            spacial_shape = input_shape[1:-1]
        else:
            spacial_shape = input_shape[2:]
        if len(self.kernel.shape) != len(spacial_shape):
            raise ValueError(
                "Kernel shape must have the same length as input, but received "
                f"kernel of shape {self.kernel.shape} and "
                f"input of shape {input_shape}."
            )
        if (
            self.dilations is not None
            and len(self.dilations) != 1
            and len(self.dilations) != len(spacial_shape)
        ):
            raise ValueError(
                "Dilation must be None, scalar or tuple/list of length of "
                "inputs' spacial shape, but received "
                f"`dilations={self.dilations}` and input of shape {input_shape}."
            )
        spacial_shape = np.array(spacial_shape)
        kernel_spacial_shape = np.array(self.kernel.shape[:2])
        dilations = np.array(self.dilations)
        if self.padding == "valid":
            output_spacial_shape = (
                np.floor(
                    (spacial_shape - dilations * (kernel_spacial_shape - 1) - 1)
                    / self.strides
                )
                + 1
            )
            negative_in_shape = np.all(output_spacial_shape < 0)
            if negative_in_shape:
                raise ValueError(
                    "Computed output size would be negative. Received "
                    f"`inputs shape={inputs.shape}`, "
                    f"`kernel spacial size={self.kernel.size}`, "
                    f"`dilations={self.dilations}`."
                )
        elif self.padding == "same":
            output_spacial_shape = (
                np.floor((spacial_shape - 1) / self.strides) + 1
            )

        output_channels = self.kernel.shape[-1] * self.kernel.shape[-2]
        if self.data_format == "channels_last":
            output_shape = (
                [input_shape[0]] + output_spacial_shape + [output_channels]
            )
        else:
            output_shape = [
                input_shape[0],
                output_channels,
            ] + output_spacial_shape
        return KerasTensor(output_shape, dtype=inputs.dtype)


def depthwise_conv(
    x,
    kernel,
    strides,
    padding="valid",
    data_format="channels_last",
    dilations=None,
):
    if any_symbolic_tensors((x,)):
        return DepthwiseConv(
            kernel, strides, padding, data_format, dilations
        ).symbolic_call(x)
    return backend.nn.depthwise_conv(
        x,
        kernel,
        strides,
        padding,
        data_format,
        dilations,
    )


class SeparableConv(Operation):
    def __init__(
        self,
        depthwise_kernel,
        pointwise_kernel,
        strides,
        padding="valid",
        data_format="channels_last",
        dilations=None,
    ):
        super().__init__()
        self.depthwise_kernel = depthwise_kernel
        self.pointwise_kernel = pointwise_kernel
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilations = dilations

    def call(self, inputs):
        return backend.nn.separable_conv(
            inputs,
            self.depthwise_kernel,
            self.pointwise_kernel,
            self.strides,
            self.padding,
            self.data_format,
            self.dilations,
        )

    def compute_output_spec(self, inputs):
        output_shape = depthwise_conv(
            inputs,
            self.depthwise_kernel,
            self.strides,
            self.padding,
            self.data_format,
            self.dilations,
        ).shape
        if self.data_format == "channels_last":
            output_shape[-1] = self.pointwise_kernel.shape[-1]
        else:
            output_shape[1] = self.pointwise_kernel.shape[-1]
        return KerasTensor(output_shape, dtype=inputs.dtype)


def separable_conv(
    x,
    depthwise_kernel,
    pointwise_kernel,
    strides,
    padding="valid",
    data_format="channels_last",
    dilations=None,
):
    if any_symbolic_tensors((x,)):
        return SeparableConv(
            depthwise_kernel,
            pointwise_kernel,
            strides,
            padding,
            data_format,
            dilations,
        ).symbolic_call(x)
    return backend.nn.separable_conv(
        x,
        depthwise_kernel,
        pointwise_kernel,
        strides,
        padding,
        data_format,
        dilations,
    )


class ConvTranspose(Operation):
    def __init__(
        self,
        kernel,
        strides,
        output_padding=None,
        padding="same",
        data_format="channels_last",
        dilations=1,
    ):
        super().__init__()
        self.kernel = kernel
        self.strides = strides
        self.output_padding = output_padding
        self.padding = padding
        self.data_format = data_format
        self.dilations = dilations

    def call(self, inputs):
        return backend.nn.conv_transpose(
            inputs,
            self.kernel,
            self.strides,
            self.output_padding,
            self.padding,
            self.data_format,
            self.dilations,
        )

    def compute_output_spec(self, inputs):
        return KerasTensor([], dtype=inputs.dtype)


def conv_transpose(
    inputs,
    kernel,
    strides,
    output_padding=None,
    padding="same",
    data_format="channels_last",
    dilations=1,
):
    if any_symbolic_tensors((inputs,)):
        return Conv(
            kernel, strides, output_padding, padding, data_format, dilations
        ).symbolic_call(inputs)
    return backend.nn.conv_transpose(
        inputs, kernel, strides, output_padding, padding, data_format, dilations
    )
