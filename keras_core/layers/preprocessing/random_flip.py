import numpy as np

from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.layers.layer import Layer
from keras_core.ops import convert_to_numpy
from keras_core.ops import convert_to_tensor
from keras_core.ops import shape
from keras_core.utils import backend_utils
from keras_core.utils.module_utils import tensorflow as tf

HORIZONTAL = "horizontal"
VERTICAL = "vertical"
HORIZONTAL_AND_VERTICAL = "horizontal_and_vertical"


@keras_core_export("keras_core.layers.RandomFlip")
class RandomFlip(Layer):
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
        if not tf.available:
            raise ImportError(
                "Layer RandomFlip requires TensorFlow. "
                "Install it via `pip install tensorflow`."
            )

        super().__init__(name=name, **kwargs)

        self.mode = mode
        if mode == HORIZONTAL:
            self.horizontal = True
            self.vertical = False
        elif mode == VERTICAL:
            self.horizontal = False
            self.vertical = True
        elif mode == HORIZONTAL_AND_VERTICAL:
            self.horizontal = True
            self.vertical = True
        else:
            raise ValueError(
                f"RandomFlip layer {self.name} received an unknown mode "
                f"argument {mode}"
            )

        self.seed = seed or backend.random.make_default_seed()

        self.supports_jit = False
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

    def call(self, inputs, training=True):
        if not isinstance(inputs, (tf.Tensor, np.ndarray, list, tuple)):
            inputs = convert_to_tensor(convert_to_numpy(inputs))

        inputs = backend.cast(inputs, self.compute_dtype)
        if training:
            outputs = self._random_flipped_inputs(inputs)
        else:
            outputs = inputs

        if (backend.backend() != "tensorflow" and not backend_utils.in_tf_graph()):
            outputs = convert_to_tensor(outputs)
        return outputs

    def _random_flipped_inputs(self, inputs):
        flipped_outputs = inputs

        if self.horizontal:
            flipped_outputs = tf.image.stateless_random_flip_left_right(flipped_outputs, seed=[self.seed, 0])

        if self.vertical:
            flipped_outputs = tf.image.stateless_random_flip_up_down(flipped_outputs, seed=[self.seed, 0])

        flipped_outputs.set_shape(shape(inputs))
        return flipped_outputs

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)

    def get_config(self):
        config = {
            "mode": self.mode,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return {**base_config, **config}
