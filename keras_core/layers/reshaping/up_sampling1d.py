import math

from keras_core import operations as ops
from keras_core.layers.input_spec import InputSpec
from keras_core.layers.layer import Layer


class UpSampling1D(Layer):
    """Upsampling layer for 1D inputs.

    Repeats each temporal step `size` times along the time axis.

    Examples:

    >>> input_shape = (2, 2, 3)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> print(x)
    [[[ 0  1  2]
      [ 3  4  5]]
     [[ 6  7  8]
      [ 9 10 11]]]
    >>> y = keras.layers.UpSampling1D(size=2)(x)
    >>> print(y)
    tf.Tensor(
      [[[ 0  1  2]
        [ 0  1  2]
        [ 3  4  5]
        [ 3  4  5]]
       [[ 6  7  8]
        [ 6  7  8]
        [ 9 10 11]
        [ 9 10 11]]], shape=(2, 4, 3), dtype=int64)

    Args:
      size: Integer. Upsampling factor.

    Input shape:
      3D tensor with shape: `(batch_size, steps, features)`.

    Output shape:
      3D tensor with shape: `(batch_size, upsampled_steps, features)`.
    """

    def __init__(self, size=2, **kwargs):
        super().__init__(**kwargs)
        self.size = int(size)
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        size = (
            self.size * input_shape[1] if input_shape[1] is not None else None
        )
        return [input_shape[0], size, input_shape[2]]

    def call(self, inputs):
        output = ops.repeat_elements(inputs, self.size, axis=1)
        return output

    def get_config(self):
        config = {"size": self.size}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))