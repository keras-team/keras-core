import numpy as np
from absl.testing import parameterized

from keras_core import layers
from keras_core import testing


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