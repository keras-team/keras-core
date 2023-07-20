from keras_core import ops
from keras_core.api_export import keras_core_export
from keras_core.layers.input_spec import InputSpec
from keras_core.layers.layer import Layer


@keras_core_export("keras_core.layers.Permute")
class Permute(Layer):
    """Permutes the dimensions of the input according to a given pattern.

    Useful e.g. connecting RNNs and convnets.

    Args:
        dims: Tuple of integers. Permutation pattern does not include the
            batch dimension. Indexing starts at 1.
            For instance, `(2, 1)` permutes the first and second dimensions
            of the input.

    Input shape:
        Arbitrary.

    Output shape:
        Same as the input shape, but with the dimensions re-ordered according
        to the specified pattern.

    Example:

    >>> x = keras_core.Input(shape=(10, 64))
    >>> y = keras_core.layers.Permute((2, 1))(x)
    >>> y.shape
    (None, 64, 10)
    """

    def __init__(self, dims, **kwargs):
        super().__init__(**kwargs)
        self.dims = tuple(dims)
        if sorted(dims) != list(range(1, len(dims) + 1)):
            raise ValueError(
                "Invalid permutation argument `dims` for Permute Layer. "
                "The set of indices in `dims` must be consecutive and start "
                f"from 1. Received dims={dims}"
            )
        self.input_spec = InputSpec(ndim=len(self.dims) + 1)

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0]]
        for dim in self.dims:
            output_shape.append(input_shape[dim])
        return tuple(output_shape)

    def call(self, inputs):
        return ops.transpose(inputs, axes=(0,) + self.dims)

    def get_config(self):
        config = {"dims": self.dims}
        base_config = super().get_config()
        return {**base_config, **config}
