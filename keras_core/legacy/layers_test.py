import unittest

import numpy as np

from keras_core.legacy.layers import (
    AlphaDropout,
    RandomHeight,
    RandomWidth,
    ThresholdedReLU,
)


class TestAlphaDropout(unittest.TestCase):
    def test_alpha_dropout(self):
        layer = AlphaDropout(rate=0.2)
        data = np.ones((10, 10))
        result = layer(data, training=True)
        self.assertTrue((result.numpy() <= 1.0).all())
        self.assertTrue((result.numpy() >= -0.5).all())

    def test_alpha_dropout_test_phase(self):
        layer = AlphaDropout(rate=0.2)
        data = np.ones((10, 10))
        result = layer(data, training=False)
        self.assertTrue((result.numpy() == 1.0).all())


class TestRandomHeight(unittest.TestCase):
    def test_random_height(self):
        layer = RandomHeight(factor=0.2)
        data = np.ones((10, 64, 64, 3))
        result = layer(data, training=True)
        self.assertEqual(result.shape, data.shape)

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
        self.assertEqual(result.shape, data.shape)

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
        expected_result = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_equal(result.numpy(), expected_result)

    def test_thresholded_relu_config(self):
        layer = ThresholdedReLU(theta=0.5)
        config = layer.get_config()
        self.assertEqual(config["theta"], 0.5)
