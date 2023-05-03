"""Keras base class for convolution layers."""


from keras_core import operations as ops
from keras_core.backend import image_data_format
from keras_core.layers.input_spec import InputSpec
from keras_core.layers.layer import Layer
from keras_core import activations
from keras_core import constraints
from keras_core import initializers
from keras_core import regularizers


class Conv(Layer):
    """Abstract N-D convolution layer (private, used as implementation base).

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    Note: layer attributes cannot be modified after the layer has been called
    once (except the `trainable` attribute).

    Args:
      rank: An integer, the rank of the convolution, e.g. "2" for 2D
        convolution.
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution). Could be "None", eg in the case of
        depth wise convolution.
      kernel_size: An integer or tuple/list of n integers, specifying the
        length of the convolution window.
      strides: An integer or tuple/list of n integers,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"`,  `"same"`, or `"causal"` (case-insensitive).
        `"valid"` means no padding. `"same"` results in padding with zeros
        evenly to the left/right or up/down of the input such that output has
        the same height/width dimension as the input. `"causal"` results in
        causal (dilated) convolutions, e.g. `output[t]` does not depend on
        `input[t+1:]`.
      data_format: A string, one of `channels_last` (default) or
        `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch_size, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch_size, channels, ...)`.
      dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      groups: A positive integer specifying the number of groups in which the
        input is split along the channel axis. Each group is convolved
        separately with `filters / groups` filters. The output is the
        concatenation of all the `groups` results along the channel axis.
        Input channels and `filters` must both be divisible by `groups`.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: An initializer for the convolution kernel. If None,
        the default initializer (glorot_uniform) will be used.
      bias_initializer: An initializer for the bias vector. If None, the default
        initializer (zeros) will be used.
      kernel_regularizer: Optional regularizer for the convolution kernel.
      bias_regularizer: Optional regularizer for the bias vector.
      activity_regularizer: Optional regularizer function for the output.
      kernel_constraint: Optional projection function to be applied to the
          kernel after being updated by an `Optimizer` (e.g. used to implement
          norm constraints or value constraints for layer weights). The function
          must take as input the unprojected variable and must return the
          projected variable (which must have the same shape). Constraints are
          not safe to use when doing asynchronous distributed training.
      bias_constraint: Optional projection function to be applied to the
          bias after being updated by an `Optimizer`.
    """

    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,
        dilation_rate=1,
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        **kwargs,
    ):
        super().__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs,
        )
        self.rank = rank
        self.filters = filters
        self.groups = groups or 1
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = (
            image_data_format() if data_format is None else data_format
        )
        self.dilation_rate = dilation_rate
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=self.rank + 2)
        self.data_format = self.data_format

        self._validate_init()

    def _validate_init(self):
        if self.filters is not None and self.filters <= 0:
            raise ValueError(
                "Invalid value for argument `filters`. Expected a strictly "
                f"positive value. Received filters={self.filters}."
            )

        if self.filters is not None and self.filters % self.groups != 0:
            raise ValueError(
                "The number of filters must be evenly divisible by the "
                f"number of groups. Received: groups={self.groups}, "
                f"filters={self.filters}."
            )

        if not all(self.kernel_size):
            raise ValueError(
                "The argument `kernel_size` cannot contain 0(s). Received: %s"
                % (self.kernel_size,)
            )

        if not all(self.strides):
            raise ValueError(
                "The argument `strides` cannot contains 0(s). Received: %s"
                % (self.strides,)
            )

        if max(self.strides) > 1 and max(self.dilation_rate) > 1:
            raise ValueError(
                "`strides > 1` not supported in conjunction with "
                f"`dilation_rate > 1`. Received: strides={self.strides} and "
                f"dilation_rate={self.dilation_rate}"
            )
