from keras_core import activations
from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.layers.layer import Layer


@keras_core_export("keras_core.layers.LeakyReLU")
class Softmax(Layer):
    """Softmax activation function.

    Example without mask:

    >>> inp = np.asarray([1., 2., 1.])
    >>> layer = tf.keras.layers.Softmax()
    >>> layer(inp).numpy()
    array([0.21194157, 0.5761169 , 0.21194157], dtype=float32)
    >>> mask = np.asarray([True, False, True], dtype=bool)
    >>> layer(inp, mask).numpy()
    array([0.5, 0. , 0.5], dtype=float32)

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as the input.

    Args:
        axis: Integer, or list of Integers, axis along which the softmax
            normalization is applied.
    Call arguments:
        inputs: The inputs, or logits to the softmax layer.
        mask: A boolean mask of the same shape as `inputs`. The mask
            specifies 1 to keep and 0 to mask. Defaults to `None`.


    Returns:
        Softmaxed output with the same shape as `inputs`.
    """

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs, mask=None):
        if mask is not None:
            adder = (1.0 - backend.cast(mask, inputs.dtype)) * (-1e9)
            inputs += adder
        if isinstance(self.axis, (tuple, list)):
            if len(self.axis) > 1:
                return backend.numpy.exp(
                    inputs
                    - backend.math.log_sum_exp(
                        inputs, axis=self.axis, keepdims=True
                    )
                )
            else:
                return activations.softmax(inputs, axis=self.axis[0])
        return activations.softmax(inputs, axis=self.axis)

    def get_config(self):
        config = {"axis": self.axis}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
