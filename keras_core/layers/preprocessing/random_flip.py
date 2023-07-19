from keras_core.api_export import keras_core_export
from keras_core.layers.preprocessing.tf_data_layer import TFDataLayer
from keras_core.random.seed_generator import SeedGenerator

HORIZONTAL = "horizontal"
VERTICAL = "vertical"
HORIZONTAL_AND_VERTICAL = "horizontal_and_vertical"


@keras_core_export("keras_core.layers.RandomFlip")
class RandomFlip(TFDataLayer):
    """A preprocessing layer which randomly flips images during training.

    This layer will flip the images horizontally and or vertically based on the
    `mode` attribute. During inference time, the output will be identical to
    input. Call the layer with `training=True` to flip the input.
    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype.
    By default, the layer will output floats.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.

    Args:
        mode: String indicating which flip mode to use. Can be `"horizontal"`,
            `"vertical"`, or `"horizontal_and_vertical"`. `"horizontal"` is a
            left-right flip and `"vertical"` is a top-bottom flip. Defaults to
            `"horizontal_and_vertical"`
        seed: Integer. Used to create a random seed.
        **kwargs: Base layer keyword arguments, such as
            `name` and `dtype`.
    """

    def __init__(
        self, mode=HORIZONTAL_AND_VERTICAL, seed=None, name=None, **kwargs
    ):
        super().__init__(name=name)
        self.seed = seed
        self.generator = SeedGenerator(seed)
        self.mode = mode
        self.supports_jit = False
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

    def _horizontal(self):
        return self.mode == HORIZONTAL or self.mode == HORIZONTAL_AND_VERTICAL

    def _vertical(self):
        return self.mode == VERTICAL or self.mode == HORIZONTAL_AND_VERTICAL

    def call(self, inputs, training=True):
        inputs = self.backend.cast(inputs, self.compute_dtype)
        if training:
            flipped_outputs = inputs
            seed_generator = self._get_seed_generator()
            if self._horizontal():
                flipped_outputs = self.backend.cond(
                    self.backend.random.uniform(shape=(), seed=seed_generator)
                    <= 0.5,
                    lambda: self.backend.numpy.flip(flipped_outputs, axis=-2),
                    lambda: flipped_outputs,
                )
            if self._vertical():
                flipped_outputs = self.backend.cond(
                    self.backend.random.uniform(shape=(), seed=seed_generator)
                    <= 0.5,
                    lambda: self.backend.numpy.flip(flipped_outputs, axis=-3),
                    lambda: flipped_outputs,
                )
            return flipped_outputs
        else:
            return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"seed": self.seed, "mode": self.mode})
        return config
