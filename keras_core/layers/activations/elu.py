from keras_core import activations
from keras_core.api_export import keras_core_export
from keras_core.layers.layer import Layer


@keras_core_export("keras_core.layers.ELU")
class ELU(Layer):
    """Applies an Exponential Linear Unit function to an output.

    Args:
        **kwargs: Base layer keyword arguments, such as
            `name` and `dtype`.

    Example:

    >>> layer = keras_core.layers.ELU()
    >>> layer([-3.0, -1.0, 0.0, 2.0])
    [0.0, 0.0, 0.0, 2.0]
    >>> layer = keras_core.layers.Activation(keras_core.activations.relu)
    >>> layer([-3.0, -1.0, 0.0, 2.0])
    [0.0, 0.0, 0.0, 2.0]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return activations.elu(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape
