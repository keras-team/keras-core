import numpy as np

from keras_core import operations as ops
from keras_core.api_export import keras_core_export
from keras_core.backend import random
from keras_core.layers.layer import Layer


@keras_core_export("keras_core.layers.RandomContrast")
class RandomContrast(Layer):
    """A preprocessing layer which randomly adjusts contrast during training.

    This layer will randomly adjust the contrast of an image or images
    by a random factor. Contrast is adjusted independently
    for each channel of each image during training.

    For each channel, this layer computes the mean of the image pixels in the
    channel and then adjusts each component `x` of each pixel to
    `(x - mean) * contrast_factor + mean`.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    in integer or floating point dtype.
    By default, the layer will output floats.
    The output value will be clipped to the range `[0, 255]`, the valid
    range of RGB colors.

    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.

    Args:
        factor: a positive float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound.
            When represented as a single float, lower = upper.
            The contrast factor will be randomly picked between
            `[1.0 - lower, 1.0 + upper]`. For any pixel x in the channel,
            the output will be `(x - mean) * factor + mean`
            where `mean` is the mean value of the channel.
        seed: Integer. Used to create a random seed.
    """

    def __init__(self, factor, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        if isinstance(factor, (tuple, list)):
            self.lower = factor[0]
            self.upper = factor[1]
        else:
            self.lower = self.upper = factor
        if self.lower < 0.0 or self.upper < 0.0 or self.lower > 1.0:
            raise ValueError(
                "`factor` argument cannot have negative values or values "
                "greater than 1."
                f"Received: factor={factor}"
            )
        if seed is not None:
            self.seed = seed
        else:
            self.seed = random.draw_seed(random.SeedGenerator(seed))

    def call(self, inputs, training=True):
        inputs = ops.cast(inputs, self.compute_dtype)

        def random_contrasted_inputs(inputs):

            np.random.seed(self.seed)
            factor = np.random.uniform(
                low=1.0 - self.lower, high=1.0 + self.upper
            )

            outputs = self._adjust_constrast(inputs, factor)
            outputs = ops.clip(outputs, 0, 255)
            ops.reshape(outputs, inputs.shape)
            return outputs

        if training:
            return random_contrasted_inputs(inputs)
        else:
            return inputs

    def _adjust_constrast(self, inputs, contrast_factor):
        # For each channel, this Op computes the mean of the image pixels in the
        # channel and then adjusts each component `x` of each pixel to
        # `(x - mean) * contrast_factor + mean`.

        # `contrast_factor` must be in the interval `(-inf, inf)`.
        # reduce mean on height
        inp_mean = np.mean(inputs, axis=-3, keepdims=True)
        # reduce mean on width
        inp_mean = np.mean(inp_mean, axis=-2, keepdims=True)

        outputs = (inputs - inp_mean) * contrast_factor + inp_mean
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "factor": self.factor,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
