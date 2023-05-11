from absl.testing import parameterized

from keras_core import layers
from keras_core import testing
from keras_core import backend
import numpy as np
from keras_core import utils


class RandomFlipTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("random_flip_horizontal", "horizontal"),
        ("random_flip_vertical", "vertical"),
        ("random_flip_both", "horizontal_and_vertical"),
    )
    def test_random_flip(self, mode):
        self.run_layer_test(
            layers.RandomFlip,
            init_kwargs={
                "mode": mode,
            },
            input_shape=(2, 3, 4),
            expected_output_shape=(2, 3, 4),
            supports_masking=False,
        )

    def test_random_flip_horizontal(self):
        utils.set_random_seed(1337)
        self.run_layer_test(
            layers.RandomFlip,
            init_kwargs={
                "mode": "vertical",
                "seed": 1234
            },
            input_data=np.asarray([[[2, 3, 4], [5, 6, 7]]]),
            expected_output=backend.convert_to_tensor([[[4, 3, 2], [7, 6, 5]]]),
            supports_masking=False,
        )
