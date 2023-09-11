import unittest

import numpy as np

from keras_core.legacy.layers import AlphaDropout
from keras_core.legacy.layers import RandomHeight
from keras_core.legacy.layers import RandomWidth
from keras_core.legacy.layers import ThresholdedReLU


def _get_theoretical_min_value(rate):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    alpha_p = -alpha * scale
    a = ((1 - rate) * (1 + rate * alpha_p**2)) ** -0.5
    b = -a * alpha_p * rate
    return a * alpha_p + b


class TestAlphaDropout(unittest.TestCase):
    def test_alpha_dropout_no_nan(self):
        layer = AlphaDropout(rate=0.2)
        data = np.ones((10, 10))
        result = layer(data, training=True)
        self.assertFalse(np.isnan(result.numpy()).any())

    def test_alpha_dropout_no_inf(self):
        layer = AlphaDropout(rate=0.2)
        data = np.ones((10, 10))
        result = layer(data, training=True)
        self.assertFalse(np.isinf(result.numpy()).any())

    def test_alpha_dropout_value_range(self):
        layer = AlphaDropout(rate=0.2)
        data = np.ones((10, 10))
        result = layer(data, training=True)

        theoretical_min = _get_theoretical_min_value(0.2)
        self.assertTrue((result >= theoretical_min).all())

    def test_alpha_dropout_test_phase(self):
        layer = AlphaDropout(rate=0.2)
        data = np.ones((10, 10))
        result = layer(data, training=False)
        self.assertTrue((result == 1.0).all())


class TestRandomHeight(unittest.TestCase):
    def test_random_height(self):
        layer = RandomHeight(factor=0.2)
        data = np.ones((10, 64, 64, 3))
        result = layer(data, training=True)
        self.assertTrue(0.8 * 64 <= result.shape[1] <= 1.2 * 64)

    def test_random_height_test_phase(self):
        layer = RandomHeight(factor=0.2)
        data = np.ones((10, 64, 64, 3))
        result = layer(data, training=False)
        self.assertTrue((result.numpy() == 1.0).all())


class TestRandomWidth(unittest.TestCase):
    def test_random_width(self):
        layer = RandomWidth(factor=0.2)
        data = np.ones((10, 64, 64, 3))
        result = layer(data, training=True)
        self.assertTrue(0.8 * 64 <= result.shape[2] <= 1.2 * 64)

    def test_random_width_test_phase(self):
        layer = RandomWidth(factor=0.2)
        data = np.ones((10, 64, 64, 3))
        result = layer(data, training=False)
        self.assertTrue((result.numpy() == 1.0).all())


class TestThresholdedReLU(unittest.TestCase):
    def test_thresholded_relu(self):
        layer = ThresholdedReLU(theta=0.5)
        data = np.array([-0.5, 0.5, 1.0])
        result = layer(data)
        expected_result = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_equal(result.numpy(), expected_result)

    def test_thresholded_relu_config(self):
        layer = ThresholdedReLU(theta=0.5)
        config = layer.get_config()
        self.assertEqual(config["theta"], 0.5)
