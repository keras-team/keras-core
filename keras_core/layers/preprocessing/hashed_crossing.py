import tensorflow as tf

from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.layers.layer import Layer


@keras_core_export("keras_core.layers.HashedCrossing")
class HashedCrossing(Layer):
    """A preprocessing layer which crosses features using the "hashing trick".

    This layer performs crosses of categorical features using the "hashing
    trick". Conceptually, the transformation can be thought of as:
    `hash(concatenate(features)) % num_bins.

    This layer currently only performs crosses of scalar inputs and batches of
    scalar inputs. Valid input shapes are `(batch_size, 1)`, `(batch_size,)` and
    `()`.

    **Note:** This layer wraps `tf.keras.layers.HashedCrossing`. It cannot
    be used as part of the compiled computation graph of a model with
    any backend other than TensorFlow.
    It can however be used with any backend when running eagerly.
    It can also always be used as part of an input preprocessing pipeline
    with any backend (outside the model itself), which is how we recommend
    to use this layer.

    Args:
        num_bins: Number of hash bins.
        output_mode: Specification for the output of the layer. Values can be
            `"int"`, or `"one_hot"` configuring the layer as follows:
            - `"int"`: Return the integer bin indices directly.
            - `"one_hot"`: Encodes each individual element in the input into an
                array the same size as `num_bins`, containing a 1 at the input's
                bin index. Defaults to `"int"`.
        sparse: Boolean. Only applicable to `"one_hot"` mode and only valid
            when using the TensorFlow backend. If `True`, returns
            a `SparseTensor` instead of a dense `Tensor`. Defaults to `False`.
        **kwargs: Keyword arguments to construct a layer.

    Examples:

    **Crossing two scalar features.**

    >>> layer = keras_core.layers.HashedCrossing(
    ...     num_bins=5)
    >>> feat1 = np.array(['A', 'B', 'A', 'B', 'A'])
    >>> feat2 = np.array([101, 101, 101, 102, 102])
    >>> layer((feat1, feat2))
    array([1, 4, 1, 1, 3])

    **Crossing and one-hotting two scalar features.**

    >>> layer = keras_core.layers.HashedCrossing(
    ...     num_bins=5, output_mode='one_hot')
    >>> feat1 = np.array(['A', 'B', 'A', 'B', 'A'])
    >>> feat2 = np.array([101, 101, 101, 102, 102])
    >>> layer((feat1, feat2))
    array([[0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 1.],
            [0., 1., 0., 0., 0.],
            [0., 1., 0., 0., 0.],
            [0., 0., 0., 1., 0.]], dtype=float32)
    """

    def __init__(
        self,
        num_bins,
        output_mode="int",
        sparse=False,
        name=None,
        dtype=None,
        **kwargs,
    ):
        if output_mode == "int" and dtype is None:
            dtype = "int64"
        super().__init__(name=name, dtype=dtype)
        if sparse and backend.backend() != "tensorflow":
            raise ValueError(
                "`sparse` can only be set to True with the "
                "TensorFlow backend."
            )
        self.layer = tf.keras.layers.HashedCrossing(
            num_bins=num_bins,
            output_mode=output_mode,
            sparse=sparse,
            name=name,
            dtype=dtype,
            **kwargs,
        )
        self.num_bins = num_bins
        self.output_mode = output_mode
        self.sparse = sparse
        self._allow_non_tensor_positional_args = True
        self._convert_input_args = False
        self.supports_jit = False

    def compute_output_shape(self, input_shape):
        return tuple(self.layer.compute_output_shape(input_shape))

    def call(self, inputs):
        outputs = self.layer.call(inputs)
        if backend.backend() != "tensorflow":
            outputs = backend.convert_to_tensor(outputs)
        return outputs

    def get_config(self):
        return {
            "num_bins": self.num_bins,
            "output_mode": self.output_mode,
            "sparse": self.sparse,
            "name": self.name,
            "dtype": self.dtype,
        }
