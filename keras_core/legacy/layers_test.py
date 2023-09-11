import unittest

import numpy as np

from keras_core.legacy.layers import (
    AlphaDropout,
    RandomHeight,
    RandomWidth,
    ThresholdedReLU,
)
from keras_core.legacy.losses import Reduction


class TestAlphaDropout(unittest.TestCase):
    def test_call(self):
        alpha_dropout = AlphaDropout(rate=0.5, seed=0)
        inputs = np.ones((2, 3, 4))
        output = alpha_dropout(inputs, training=True)
        self.assertEqual(output.shape, inputs.shape)

    def test_get_config(self):
        alpha_dropout = AlphaDropout(rate=0.5, seed=0)
        config = alpha_dropout.get_config()
        self.assertEqual(config["rate"], 0.5)
        self.assertEqual(config["seed"], 0)


class TestRandomHeight(unittest.TestCase):
    def test_call(self):
        random_height = RandomHeight(factor=0.5)
        inputs = np.ones((2, 3, 4))
        output = random_height(inputs, training=True)
        self.assertEqual(output.shape, (2, 3, 4))

    def test_get_config(self):
        random_height = RandomHeight(factor=0.5)
        config = random_height.get_config()
        self.assertEqual(config["factor"], 0.5)
        self.assertEqual(config["interpolation"], "bilinear")
        self.assertIsNone(config["seed"])


class TestRandomWidth(unittest.TestCase):
    def test_call(self):
        random_width = RandomWidth(factor=0.5)
        inputs = np.ones((2, 3, 4))
        output = random_width(inputs, training=True)
        self.assertEqual(output.shape, (2, 3, 4))

    def test_get_config(self):
        random_width = RandomWidth(factor=0.5)
        config = random_width.get_config()
        self.assertEqual(config["factor"], 0.5)
        self.assertEqual(config["interpolation"], "bilinear")
        self.assertIsNone(config["seed"])


class TestThresholdedReLU(unittest.TestCase):
    def test_call(self):
        thresholded_relu = ThresholdedReLU(theta=0.5)
        inputs = np.ones((2, 3, 4))
        output = thresholded_relu(inputs)
        self.assertEqual(output.shape, (2, 3, 4))

    def test_get_config(self):
        thresholded_relu = ThresholdedReLU(theta=0.5)
        config = thresholded_relu.get_config()
        self.assertEqual(config["theta"], 0.5)
