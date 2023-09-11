import unittest

import numpy as np
import pytest

from keras_core import backend
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
    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="Skipping this test for PyTorch backend.",
    )
    def test_alpha_dropout_no_nan(self):
        # TODO: Address the test failure for the torch backend
        layer = AlphaDropout(rate=0.2)
        data = np.ones((10, 10))
        result = layer(data, training=True)
        self.assertFalse(np.isnan(result.numpy()).any())

    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="Skipping this test for PyTorch backend.",
    )
    def test_alpha_dropout_no_inf(self):
        # TODO: Address the test failure for the torch backend
        layer = AlphaDropout(rate=0.2)
        data = np.ones((10, 10))
        result = layer(data, training=True)
        self.assertFalse(np.isinf(result.numpy()).any())

    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="Skipping this test for PyTorch backend.",
    )
    def test_alpha_dropout_value_range(self):
        # TODO: Address the test failure for the torch backend
        layer = AlphaDropout(rate=0.2)
        data = np.ones((10, 10))
        result = layer(data, training=True)
        theoretical_min = _get_theoretical_min_value(0.2)
        self.assertTrue((result.numpy() >= theoretical_min).all())

    @pytest.mark.skipif(backend.backend() != "tensorflow", reason="for all()")
    def test_alpha_dropout_test_phase_for_tf(self):
        layer = AlphaDropout(rate=0.2)
        data = np.ones((10, 10))
        result = layer(data, training=False)
        self.assertTrue((result.numpy() == 1.0).all())

    @pytest.mark.skipif(backend.backend() == "tensorflow", reason="for all()")
    def test_alpha_dropout_test_phase_for_others(self):
        layer = AlphaDropout(rate=0.2)
        data = np.ones((10, 10))
        result = layer(data, training=False)
        self.assertTrue((result == 1.0).all())


class TestRandomHeight(unittest.TestCase):
    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="Skipping this test for PyTorch backend.",
    )
    def test_random_height(self):
        # TODO: Address the test failure for the torch backend
        layer = RandomHeight(factor=0.2)
        data = np.ones((10, 64, 64, 3))
        result = layer(data, training=True)
        self.assertTrue(0.8 * 64 <= result.shape[1] <= 1.2 * 64)

    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="Skipping this test for PyTorch backend.",
    )
    def test_random_height_test_phase(self):
        # TODO: Address the test failure for the torch backend
        layer = RandomHeight(factor=0.2)
        data = np.ones((10, 64, 64, 3))
        result = layer(data, training=False)
        self.assertTrue((result.numpy() == 1.0).all())


class TestRandomWidth(unittest.TestCase):
    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="Skipping this test for PyTorch backend.",
    )
    def test_random_width(self):
        # TODO: Address the test failure for the torch backend
        layer = RandomWidth(factor=0.2)
        data = np.ones((10, 64, 64, 3))
        result = layer(data, training=True)
        self.assertTrue(0.8 * 64 <= result.shape[2] <= 1.2 * 64)

    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="Skipping this test for PyTorch backend.",
    )
    def test_random_width_test_phase(self):
        # TODO: Address the test failure for the torch backend
        layer = RandomWidth(factor=0.2)
        data = np.ones((10, 64, 64, 3))
        result = layer(data, training=False)
        self.assertTrue((result.numpy() == 1.0).all())


class TestThresholdedReLU(unittest.TestCase):
    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="Skipped for torch backend due to type issues.",
    )
    # TODO: Address the test failure for the torch backend
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
