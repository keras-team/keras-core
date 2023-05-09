import numpy as np

from keras_core import backend
from keras_core import layers
from keras_core import testing
from keras_core.layers.preprocessing import random_crop


class RandomCrop(testing.TestCase):
    def test_random_crop(self):
        self.run_layer_test(
            random_crop.RandomCrop,
            init_kwargs={
                "height" : 1,
                "width" : 1,
            },
            input_shape=(2, 3, 4),
            supports_masking=False,
        )