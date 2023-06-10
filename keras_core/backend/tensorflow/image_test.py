"""Tests for tf.distribute related functionality under tf implementation."""

import math

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from keras_core import testing
from keras_core.backend.tensorflow.image import _resize_repeat_interpolation


class ImageTest(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        [
            {"input_size": (10, 10), "target_size": (10, 10), "channels": 1},
            {"input_size": (10, 10), "target_size": (15, 15), "channels": 1},
            {"input_size": (10, 10), "target_size": (20, 20), "channels": 1},
            {"input_size": (10, 10), "target_size": (10, 10), "channels": 3},
            {"input_size": (10, 10), "target_size": (15, 15), "channels": 3},
            {"input_size": (10, 10), "target_size": (20, 20), "channels": 3},
        ]
    )
    def test_image_resize_nearest_interp(
        self, input_size, target_size, channels
    ):
        for batches in [0, 1, 5]:
            elem = math.prod(input_size) * channels
            if batches == 0:
                x = np.arange(elem).reshape(*input_size, channels)
            else:
                elem *= batches
                x = np.arange(elem).reshape(batches, *input_size, channels)

            ref_res = tf.image.resize(x, target_size, method="nearest")
            # Repeat on dimensions
            res = _resize_repeat_interpolation(x, target_size)
            self.assertEqual(ref_res.shape, res.shape)
            # For proper multiples, results should match for nearest & repeat.
            # For improper multiples, differences exist for nearest & repeat.
            check_repeat_ratio = [
                target_sz % input_sz
                for input_sz, target_sz in zip(input_size, target_size)
            ]
            if sum(check_repeat_ratio) == len(target_size):
                self.assertTrue(np.allclose(ref_res, res))
