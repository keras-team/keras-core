import numpy as np

from keras_core import backend
from keras_core import operations as ops
from keras_core.layers.input_spec import InputSpec
from keras_core.layers.layer import Layer
from keras_core.utils import argument_validation


class UpSampling2D(Layer):
    """Upsampling layer for 2D inputs.

    Repeats the rows and columns of the data
    by `size[0]` and `size[1]` respectively.

    Examples:

    >>> input_shape = (2, 2, 1, 3)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> print(x)
    [[[[ 0  1  2]]
      [[ 3  4  5]]]
     [[[ 6  7  8]]
      [[ 9 10 11]]]]
    >>> y = keras_core.layers.UpSampling2D(size=(1, 2))(x)
    >>> print(y)
    [[[[ 0  1  2]
       [ 0  1  2]]
      [[ 3  4  5]
       [ 3  4  5]]]
     [[[ 6  7  8]
       [ 6  7  8]]
      [[ 9 10 11]
       [ 9 10 11]]]]

    Args:
      size: Int, or tuple of 2 integers.
          The upsampling factors for rows and columns.
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch_size, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch_size, channels, height, width)`.
          When unspecified, uses
          `image_data_format` value found in your Keras config file at
          `~/.keras/keras.json` (if exists) else `"channels_last"`.
          Defaults to `"channels_last"`.
      interpolation: A string, one of `"bicubic"`, `"bilinear"`, `"lanczos3"`,
          `"lanczos5"`, `"nearest"`.

    Input shape:
      4D tensor with shape:
      - If `data_format` is `"channels_last"`:
            `(batch_size, rows, cols, channels)`
      - If `data_format` is `"channels_first"`:
            `(batch_size, channels, rows, cols)`

    Output shape:
      4D tensor with shape:
      - If `data_format` is `"channels_last"`:
            `(batch_size, upsampled_rows, upsampled_cols, channels)`
      - If `data_format` is `"channels_first"`:
            `(batch_size, channels, upsampled_rows, upsampled_cols)`
    """

    def __init__(
        self, size=(2, 2), data_format=None, interpolation="nearest", **kwargs
    ):
        super().__init__(**kwargs)
        self.data_format = backend.config.standardize_data_format(data_format)
        self.size = argument_validation.standardize_tuple(size, 2, "size")
        self.interpolation = interpolation.lower()
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            height = (
                self.size[0] * input_shape[2]
                if input_shape[2] is not None
                else None
            )
            width = (
                self.size[1] * input_shape[3]
                if input_shape[3] is not None
                else None
            )
            return (input_shape[0], input_shape[1], height, width)
        else:
            height = (
                self.size[0] * input_shape[1]
                if input_shape[1] is not None
                else None
            )
            width = (
                self.size[1] * input_shape[2]
                if input_shape[2] is not None
                else None
            )
            return (input_shape[0], height, width, input_shape[3])

    def call(self, inputs):
        return self._resize_images(
            inputs,
            self.size[0],
            self.size[1],
            self.data_format,
            interpolation=self.interpolation,
        )

    def get_config(self):
        config = {
            "size": self.size,
            "data_format": self.data_format,
            "interpolation": self.interpolation,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def _resize_images(
        self,
        x,
        height_factor,
        width_factor,
        data_format,
        interpolation="nearest",
    ):
        """Resizes the images contained in a 4D tensor.

        Args:
            x: Tensor or variable to resize.
            height_factor: Positive integer.
            width_factor: Positive integer.
            data_format: One of `"channels_first"`, `"channels_last"`.
            interpolation: A string, one of `"area"`, `"bicubic"`, `"bilinear"`,
            `"gaussian"`, `"lanczos3"`, `"lanczos5"`, `"mitchellcubic"`,
            `"nearest"`.

        Returns:
            A tensor.

        Raises:
            ValueError: in case of incorrect value for
            `data_format` or `interpolation`.
        """
        if data_format == "channels_first":
            rows, cols = 2, 3
        elif data_format == "channels_last":
            rows, cols = 1, 2
        else:
            raise ValueError(f"Invalid `data_format` argument: {data_format}")

        new_shape = x.shape[rows : cols + 1]
        new_shape *= np.array([height_factor, width_factor])

        if data_format == "channels_first":
            x = ops.transpose(x, [0, 2, 3, 1])
        x = ops.image.resize(x, new_shape, method=interpolation)
        if data_format == "channels_first":
            x = ops.transpose(x, [0, 3, 1, 2])

        return x
