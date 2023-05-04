from keras_core.api_export import keras_core_export
from keras_core.layers.convolutional.base_conv import BaseConv

@keras_core_export(["keras_core.layers.Conv1D", "keras_core.layers.Convolution1D"])
class Conv1D(BaseConv):
    """1D convolution layer (e.g. temporal convolution).

    This layer creates a convolution kernel that is convolved
    with the layer input over a single spatial (or temporal) dimension
    to produce a tensor of outputs.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    When using this layer as the first layer in a model,
    provide an `input_shape` argument
    (tuple of integers or `None`, e.g.
    `(10, 128)` for sequences of 10 vectors of 128-dimensional vectors,
    or `(None, 128)` for variable-length sequences of 128-dimensional vectors.

    Examples:

    >>> # The inputs are 128-length vectors with 10 timesteps, and the
    >>> # batch size is 4.
    >>> input_shape = (4, 10, 128)
    >>> x = tf.random.normal(input_shape)
    >>> y = tf.keras.layers.Conv1D(
    ... 32, 3, activation='relu',input_shape=input_shape[1:])(x)
    >>> print(y.shape)
    (4, 8, 32)

    >>> # With extended batch shape [4, 7] (e.g. weather data where batch
    >>> # dimensions correspond to spatial location and the third dimension
    >>> # corresponds to time.)
    >>> input_shape = (4, 7, 10, 128)
    >>> x = tf.random.normal(input_shape)
    >>> y = tf.keras.layers.Conv1D(
    ... 32, 3, activation='relu', input_shape=input_shape[2:])(x)
    >>> print(y.shape)
    (4, 7, 8, 32)

    Args:
      filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of a single integer,
        specifying the length of the 1D convolution window.
      strides: An integer or tuple/list of a single integer,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"`, `"same"` or `"causal"` (case-insensitive).
        `"valid"` means no padding. `"same"` results in padding with zeros
        evenly to the left/right or up/down of the input such that output has
        the same height/width dimension as the input.
        `"causal"` results in causal (dilated) convolutions, e.g. `output[t]`
        does not depend on `input[t+1:]`. Useful when modeling temporal data
        where the model should not violate the temporal order.
        See [WaveNet: A Generative Model for Raw Audio, section
          2.1](https://arxiv.org/abs/1609.03499).
      data_format: A string, one of `channels_last` (default) or
        `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch_size, width,
        channels)` while `channels_first` corresponds to inputs with shape
        `(batch_size, channels, width)`. Note that the `channels_first` format
        is currently not supported by TensorFlow on CPU.
      dilation_rate: an integer or tuple/list of a single integer, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      groups: A positive integer specifying the number of groups in which the
        input is split along the channel axis. Each group is convolved
        separately with `filters / groups` filters. The output is the
        concatenation of all the `groups` results along the channel axis.
        Input channels and `filters` must both be divisible by `groups`.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (see `keras.activations`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix
        (see `keras.initializers`). Defaults to 'glorot_uniform'.
      bias_initializer: Initializer for the bias vector
        (see `keras.initializers`). Defaults to 'zeros'.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix (see `keras.regularizers`).
      bias_regularizer: Regularizer function applied to the bias vector
        (see `keras.regularizers`).
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation")
        (see `keras.regularizers`).
      kernel_constraint: Constraint function applied to the kernel matrix
        (see `keras.constraints`).
      bias_constraint: Constraint function applied to the bias vector
        (see `keras.constraints`).

    Input shape:
      3+D tensor with shape: `batch_shape + (steps, input_dim)`

    Output shape:
      3+D tensor with shape: `batch_shape + (new_steps, filters)`
        `steps` value might have changed due to padding or strides.

    Returns:
      A tensor of rank 3 representing
      `activation(conv1d(inputs, kernel) + bias)`.

    Raises:
      ValueError: when both `strides > 1` and `dilation_rate > 1`.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format="channels_last",
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
        **kwargs
    ):
        super().__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
