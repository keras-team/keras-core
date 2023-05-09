from keras_core import initializers
from keras_core import operations as ops
from keras_core.api_export import keras_core_export
from keras_core.layers import Wrapper
from keras_core.layers.input_spec import InputSpec


@keras_core_export("keras_core.layers.SpectralNormalization")
class SpectralNormalization(Wrapper):
    """Performs spectral normalization on the weights of a target layer.

    This wrapper controls the Lipschitz constant of the weights of a layer by
    constraining their spectral norm, which can stabilize the training of GANs.

    Args:
        layer: A `keras_core.layers.Layer` instance that
            has either a `kernel` (e.g. `Conv2D`, `Dense`...)
            or an `embeddings` attribute (`Embedding` layer).
        power_iterations: int, the number of iterations during normalization.

    Examples:

    Wrap `keras_core.layers.Conv2D`:
    >>> x = np.random.rand(1, 10, 10, 1)
    >>> conv2d = SpectralNormalization(keras_core.layers.Conv2D(2, 2))
    >>> y = conv2d(x)
    >>> y.shape
    (1, 9, 9, 2)

    Wrap `keras_core.layers.Dense`:
    >>> x = np.random.rand(1, 10, 10, 1)
    >>> dense = SpectralNormalization(keras_core.layers.Dense(10))
    >>> y = dense(x)
    >>> y.shape
    (1, 10, 10, 10)

    Reference:

    - [Spectral Normalization for GAN](https://arxiv.org/abs/1802.05957).
    """

    def __init__(self, layer, power_iterations=1, **kwargs):
        super().__init__(layer, **kwargs)
        if power_iterations <= 0:
            raise ValueError(
                "`power_iterations` should be greater than zero. Received: "
                f"`power_iterations={power_iterations}`"
            )
        self.power_iterations = power_iterations

    def build(self, input_shape):
        super().build(input_shape)
        self.input_spec = InputSpec(shape=[None] + list(input_shape[1:]))

        if hasattr(self.layer, "kernel"):
            self.kernel = self.layer.kernel
        elif hasattr(self.layer, "embeddings"):
            self.kernel = self.layer.embeddings
        else:
            raise ValueError(
                f"{type(self.layer).__name__} object has no attribute 'kernel' "
                "nor 'embeddings'"
            )

        self.kernel_shape = self.kernel.shape

        self.vector_u = self.add_weight(
            shape=(1, self.kernel_shape[-1]),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name="vector_u",
            dtype=self.kernel.dtype,
        )

    def call(self, inputs, training=False):
        if training:
            self.normalize_weights()

        output = self.layer(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def normalize_weights(self):
        """Generate spectral normalized weights.

        This method will update the value of `self.kernel` with the
        spectral normalized value, so that the layer is ready for `call()`.
        """

        weights = ops.reshape(self.kernel, [-1, self.kernel_shape[-1]])
        vector_u = self.vector_u

        # check for zeroes weights
        if not all([w == 0.0 for w in weights]):
            for _ in range(self.power_iterations):
                vector_v = self._l2_normalize(
                    ops.matmul(vector_u, ops.transpose(weights))
                )
                vector_u = self._l2_normalize(ops.matmul(vector_v, weights))
            # vector_u = tf.stop_gradient(vector_u)
            # vector_v = tf.stop_gradient(vector_v)
            sigma = ops.matmul(
                ops.matmul(vector_v, weights), ops.transpose(vector_u)
            )
            self.vector_u.assign(ops.cast(vector_u, self.vector_u.dtype))
            self.kernel.assign(
                ops.cast(
                    ops.reshape(self.kernel / sigma, self.kernel_shape),
                    self.kernel.dtype,
                )
            )

    def _l2_normalize(self, x):
        square_sum = ops.sum(ops.square(x), keepdims=True)
        x_inv_norm = 1 / ops.sqrt(ops.maximum(square_sum, 1e-12))
        return ops.multiply(x, x_inv_norm)

    def get_config(self):
        config = {"power_iterations": self.power_iterations}
        base_config = super().get_config()
        return {**base_config, **config}
