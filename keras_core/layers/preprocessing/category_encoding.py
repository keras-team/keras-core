import tensorflow as tf

from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.layers.layer import Layer
from keras_core.utils import backend_utils


def dense_bincount(inputs, depth, binary_output, dtype, count_weights=None):
    """Apply binary or count encoding to an input."""
    return tf.math.bincount(
        inputs,
        weights=count_weights,
        minlength=depth,
        maxlength=depth,
        dtype=dtype,
        axis=-1,
        binary_output=binary_output,
    )


def encode_categorical_inputs(
    inputs,
    output_mode,
    depth,
    dtype="float32",
    count_weights=None,
):
    """Encodes categoical inputs according to output_mode."""

    original_shape = inputs.shape

    # TODO(b/190445202): remove output rank restriction.
    if inputs.shape.rank > 2:
        raise ValueError(
            f"Maximum supported input rank is 2. "
            f"Received output_mode={output_mode} and input shape "
            f"{original_shape}."
        )

    # In all cases, we should uprank scalar input to a single sample.
    if inputs.shape.rank == 0:
        inputs = tf.expand_dims(inputs, -1)

    # One hot will unprank only if the final output dimension is not already 1.
    if output_mode == "one_hot":
        inputs = tf.reshape(inputs, [-1, 1])

    binary_output = output_mode in ("one_hot", "multi_hot")

    bincounts = dense_bincount(
        inputs, depth, binary_output, dtype, count_weights
    )

    if inputs.shape.rank == 1:
        bincounts.set_shape(tf.TensorShape((depth,)))
    else:
        batch_size = tf.shape(inputs)[0]
        bincounts.set_shape(tf.TensorShape((batch_size, depth)))
    return bincounts


@keras_core_export("keras_core.layers.CategoryEncoding")
class CategoryEncoding(Layer):
    """A preprocessing layer which encodes integer features.

    This layer provides options for condensing data into a categorical encoding
    when the total number of tokens are known in advance. It accepts integer
    values as inputs, and it outputs a dense or sparse representation of those
    inputs. For integer inputs where the total number of tokens is not known,
    use `keras_core.layers.IntegerLookup` instead.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

    Examples:

    **One-hot encoding data**

    >>> layer = keras_core.layers.CategoryEncoding(
    ...           num_tokens=4, output_mode="one_hot")
    >>> layer([3, 2, 0, 1])
    array([[0., 0., 0., 1.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.]]>

    **Multi-hot encoding data**

    >>> layer = keras_core.layers.CategoryEncoding(
    ...           num_tokens=4, output_mode="multi_hot")
    >>> layer([[0, 1], [0, 0], [1, 2], [3, 1]])
    array([[1., 1., 0., 0.],
            [1., 0., 0., 0.],
            [0., 1., 1., 0.],
            [0., 1., 0., 1.]]>

    **Using weighted inputs in `"count"` mode**

    >>> layer = keras_core.layers.CategoryEncoding(
    ...           num_tokens=4, output_mode="count")
    >>> count_weights = np.array([[.1, .2], [.1, .1], [.2, .3], [.4, .2]])
    >>> layer([[0, 1], [0, 0], [1, 2], [3, 1]], count_weights=count_weights)
      array([[0.1, 0.2, 0. , 0. ],
             [0.2, 0. , 0. , 0. ],
             [0. , 0.2, 0.3, 0. ],
             [0. , 0.2, 0. , 0.4]]>

    Args:
        num_tokens: The total number of tokens the layer should support. All
            inputs to the layer must integers in the range `0 <= value <
            num_tokens`, or an error will be thrown.
        output_mode: Specification for the output of the layer.
            Values can be `"one_hot"`, `"multi_hot"` or `"count"`,
            configuring the layer as follows:
                - `"one_hot"`: Encodes each individual element in the input
                        into an array of `num_tokens` size, containing a 1
                        at the element index. If the last dimension is size
                        1, will encode on that dimension. If the last
                        dimension is not size 1, will append a new dimension
                        for the encoded output.
                - `"multi_hot"`: Encodes each sample in the input into a single
                        array of `num_tokens` size, containing a 1 for each
                        vocabulary term present in the sample. Treats the last
                        dimension as the sample dimension, if input shape is
                        `(..., sample_length)`, output shape will be
                        `(..., num_tokens)`.
                - `"count"`: Like `"multi_hot"`, but the int array contains a
                        count of the number of times the token at that index
                        appeared in the sample.
            For all output modes, currently only output up to rank 2 is
            supported. Defaults to `"multi_hot"`.

    Call arguments:
        inputs: A 1D or 2D tensor of integer inputs.
        count_weights: A tensor in the same shape as `inputs` indicating the
            weight for each sample value when summing up in `count` mode.
            Not used in `"multi_hot"` or `"one_hot"` modes.
    """

    def __init__(self, num_tokens=None, output_mode="multi_hot", **kwargs):
        super().__init__(**kwargs)

        # 'output_mode' must be one of ("count", "one_hot", "multi_hot")
        if output_mode not in ("count", "one_hot", "multi_hot"):
            raise ValueError(f"Unknown arg for output_mode: {output_mode}")

        if num_tokens is None:
            raise ValueError(
                "num_tokens must be set to use this layer. If the "
                "number of tokens is not known beforehand, use the "
                "IntegerLookup layer instead."
            )
        if num_tokens < 1:
            raise ValueError(
                f"`num_tokens` must be >= 1. Received: num_tokens={num_tokens}."
            )
        self.num_tokens = num_tokens
        self.output_mode = output_mode
        self._allow_non_tensor_positional_args = True
        self._convert_input_args = False

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        if not input_shape:
            return (self.num_tokens,)
        if self.output_mode == "one_hot" and input_shape[-1] != 1:
            return tuple(input_shape + [self.num_tokens])
        else:
            return tuple(input_shape[:-1] + [self.num_tokens])

    def get_config(self):
        config = {
            "num_tokens": self.num_tokens,
            "output_mode": self.output_mode,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def call(self, inputs, count_weights=None):
        inputs = tf.convert_to_tensor(inputs)
        inputs = tf.cast(inputs, self.compute_dtype)

        if count_weights is not None:
            if self.output_mode != "count":
                raise ValueError(
                    f"`count_weights` is not used when `output_mode` is not "
                    f"`'count'`. Received `count_weights={count_weights}`."
                )
            count_weights = tf.convert_to_tensor(
                count_weights, self.compute_dtype
            )

        depth = self.num_tokens
        max_value = tf.reduce_max(inputs)
        min_value = tf.reduce_min(inputs)

        condition = tf.logical_and(
            tf.greater(tf.cast(depth, max_value.dtype), max_value),
            tf.greater_equal(min_value, tf.cast(0, min_value.dtype)),
        )
        assertion = tf.Assert(
            condition,
            [
                f"Input values must be in the range 0 <= values < num_tokens"
                f" with num_tokens={depth}."
            ],
        )

        with tf.control_dependencies([assertion]):
            outputs = encode_categorical_inputs(
                inputs,
                output_mode=self.output_mode,
                depth=depth,
                dtype=self.compute_dtype,
                count_weights=count_weights,
            )
        if (
            backend.backend() != "tensorflow"
            and not backend_utils.in_tf_graph()
        ):
            outputs = backend.convert_to_tensor(outputs)
        return outputs
