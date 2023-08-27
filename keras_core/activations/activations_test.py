import numpy as np

from keras_core import activations
from keras_core import backend
from keras_core import testing


def _ref_softmax(values):
    m = np.max(values)
    e = np.exp(values - m)
    return e / np.sum(e)


def _ref_softplus(x):
    return np.log(np.ones_like(x) + np.exp(x))


def _ref_log_softmax(values):
    max_val = np.max(values)  # for numerical stability
    stabilized_values = values - max_val
    log_sum_exp = np.log(np.sum(np.exp(stabilized_values)))
    return stabilized_values - log_sum_exp


def _ref_leaky_relu(x, alpha=0.2):
    return x if x > 0 else alpha * x


def _ref_relu6(x):
    return min(max(0, x), 6)


def _ref_silu(x):
    return x / (1 + np.exp(-x))


class ActivationsTest(testing.TestCase):
    def test_softmax(self):
        x = np.random.random((2, 5))

        result = activations.softmax(x[np.newaxis, :])[0]
        expected = _ref_softmax(x[0])
        self.assertAllClose(result[0], expected, rtol=1e-05)

    def test_softmax_2d_axis_0(self):
        x = np.random.random((2, 5))
        result = activations.softmax(x[np.newaxis, :], axis=1)[0]
        expected = np.zeros((2, 5))
        for i in range(5):
            expected[:, i] = _ref_softmax(x[:, i])
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_softmax_3d_axis_tuple(self):
        x = np.random.random((2, 3, 5))
        result = activations.softmax(x, axis=(1, 2))
        expected = np.zeros((2, 3, 5))
        for i in range(2):
            expected[i, :, :] = _ref_softmax(x[i, :, :])
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_softmax_1d(self):
        x = np.random.random(5)
        result = activations.softmax(x)
        expected = _ref_softmax(x)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_softmax_higher_dim(self):
        x = np.random.random((2, 3, 4, 5))
        result = activations.softmax(x, axis=(2, 3))
        expected = np.zeros((2, 3, 4, 5))
        for i in range(2):
            for j in range(3):
                expected[i, j, :, :] = _ref_softmax(x[i, j, :, :])
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_softmax_higher_dim_multiple_axes(self):
        x = np.random.random((2, 3, 4, 5, 6))
        result = activations.softmax(x, axis=(2, 3, 4))
        expected = np.zeros((2, 3, 4, 5, 6))
        for i in range(2):
            for j in range(3):
                expected[i, j, :, :, :] = _ref_softmax(x[i, j, :, :, :])
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_softmax_negative_axis(self):
        x = np.random.random((2, 5))
        result = activations.softmax(x, axis=-1)
        expected = np.zeros((2, 5))
        for i in range(2):
            expected[i, :] = _ref_softmax(x[i, :])
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_temporal_softmax(self):
        x = np.random.random((2, 2, 3)) * 10
        result = activations.softmax(x[np.newaxis, :])[0]
        expected = _ref_softmax(x[0, 0])
        self.assertAllClose(result[0, 0], expected, rtol=1e-05)

    def test_log_softmax_2d_axis_0(self):
        x = np.random.random((2, 5))
        result = activations.log_softmax(x[np.newaxis, :], axis=1)[0]
        expected = np.zeros((2, 5))
        for i in range(5):
            expected[:, i] = _ref_log_softmax(x[:, i])
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_log_softmax_3d_axis_tuple(self):
        x = np.random.random((2, 3, 5))
        result = activations.log_softmax(x, axis=(1, 2))
        expected = np.zeros((2, 3, 5))
        for i in range(2):
            expected[i, :, :] = _ref_log_softmax(x[i, :, :])
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_log_softmax_1d(self):
        x = np.random.random(5)
        result = activations.log_softmax(x)
        expected = _ref_log_softmax(x)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_log_softmax_higher_dim(self):
        x = np.random.random((2, 3, 4, 5))
        result = activations.log_softmax(x, axis=(2, 3))
        expected = np.zeros((2, 3, 4, 5))
        for i in range(2):
            for j in range(3):
                expected[i, j, :, :] = _ref_log_softmax(x[i, j, :, :])
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_log_softmax_higher_dim_multiple_axes(self):
        x = np.random.random((2, 3, 4, 5, 6))
        result = activations.log_softmax(x, axis=(2, 3, 4))
        expected = np.zeros((2, 3, 4, 5, 6))
        for i in range(2):
            for j in range(3):
                expected[i, j, :, :, :] = _ref_log_softmax(x[i, j, :, :, :])
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_log_softmax_negative_axis(self):
        x = np.random.random((2, 5))
        result = activations.log_softmax(x, axis=-1)
        expected = np.zeros((2, 5))
        for i in range(2):
            expected[i, :] = _ref_log_softmax(x[i, :])
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_temporal_log_softmax(self):
        x = np.random.random((2, 2, 3)) * 10
        result = activations.log_softmax(x[np.newaxis, :])[0]
        expected = _ref_log_softmax(x[0, 0])
        self.assertAllClose(result[0, 0], expected, rtol=1e-05)

    def test_selu(self):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946

        positive_values = np.array([[1, 2]], dtype=backend.floatx())
        result = activations.selu(positive_values[np.newaxis, :])[0]
        self.assertAllClose(result, positive_values * scale, rtol=1e-05)

        negative_values = np.array([[-1, -2]], dtype=backend.floatx())
        result = activations.selu(negative_values[np.newaxis, :])[0]
        true_result = (np.exp(negative_values) - 1) * scale * alpha
        self.assertAllClose(result, true_result)

    def test_softplus(self):
        x = np.random.random((2, 5))
        result = activations.softplus(x[np.newaxis, :])[0]
        expected = _ref_softplus(x)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_softsign(self):
        def softsign(x):
            return np.divide(x, np.ones_like(x) + np.absolute(x))

        x = np.random.random((2, 5))
        result = activations.softsign(x[np.newaxis, :])[0]
        expected = softsign(x)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_sigmoid(self):
        def ref_sigmoid(x):
            if x >= 0:
                return 1 / (1 + np.exp(-x))
            else:
                z = np.exp(x)
                return z / (1 + z)

        sigmoid = np.vectorize(ref_sigmoid)

        x = np.random.random((2, 5))
        result = activations.sigmoid(x[np.newaxis, :])[0]
        expected = sigmoid(x)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_hard_sigmoid(self):
        def ref_hard_sigmoid(x):
            x = (x / 6.0) + 0.5
            z = 0.0 if x <= 0 else (1.0 if x >= 1 else x)
            return z

        hard_sigmoid = np.vectorize(ref_hard_sigmoid)
        x = np.random.random((2, 5))
        result = activations.hard_sigmoid(x[np.newaxis, :])[0]
        expected = hard_sigmoid(x)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_relu(self):
        positive_values = np.random.random((2, 5))
        result = activations.relu(positive_values[np.newaxis, :])[0]
        self.assertAllClose(result, positive_values, rtol=1e-05)

        negative_values = np.random.uniform(-1, 0, (2, 5))
        result = activations.relu(negative_values[np.newaxis, :])[0]
        expected = np.zeros((2, 5))
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_leaky_relu(self):
        leaky_relu_vectorized = np.vectorize(_ref_leaky_relu)

        # Test for negative_slope = 0.01
        # Test positive values
        positive_values = np.random.random((2, 5))
        result = activations.leaky_relu(
            positive_values[np.newaxis, :], negative_slope=0.01
        )[0]
        expected = leaky_relu_vectorized(positive_values, alpha=0.01)
        self.assertAllClose(result, expected, rtol=1e-05)

        # Test negative values
        negative_values = np.random.uniform(-1, 0, (2, 5))
        result = activations.leaky_relu(
            negative_values[np.newaxis, :], negative_slope=0.01
        )[0]
        expected = leaky_relu_vectorized(negative_values, alpha=0.01)
        self.assertAllClose(result, expected, rtol=1e-05)

        # Test for negative_slope = 0.3
        # Test positive values
        positive_values = np.random.random((2, 5))
        result = activations.leaky_relu(
            positive_values[np.newaxis, :], negative_slope=0.3
        )[0]
        expected = leaky_relu_vectorized(positive_values, alpha=0.3)
        self.assertAllClose(result, expected, rtol=1e-05)

        # Test negative values
        negative_values = np.random.uniform(-1, 0, (2, 5))
        result = activations.leaky_relu(
            negative_values[np.newaxis, :], negative_slope=0.3
        )[0]
        expected = leaky_relu_vectorized(negative_values, alpha=0.3)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_relu6(self):
        relu6_vectorized = np.vectorize(_ref_relu6)

        # Test positive values less than 6
        positive_values = np.random.uniform(0, 5.9, (2, 5))
        result = activations.relu6(positive_values[np.newaxis, :])[0]
        expected = relu6_vectorized(positive_values)
        self.assertAllClose(result, expected, rtol=1e-05)

        # Test positive values greater than 6
        positive_values_above_6 = np.random.uniform(6.1, 10, (2, 5))
        result = activations.relu6(positive_values_above_6[np.newaxis, :])[0]
        expected = relu6_vectorized(positive_values_above_6)
        self.assertAllClose(result, expected, rtol=1e-05)

        # Test negative values
        negative_values = np.random.uniform(-1, 0, (2, 5))
        result = activations.relu6(negative_values[np.newaxis, :])[0]
        expected = relu6_vectorized(negative_values)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_silu(self):
        silu_vectorized = np.vectorize(_ref_silu)

        # Test positive values
        positive_values = np.random.uniform(0, 5.9, (2, 5))
        result = activations.silu(positive_values[np.newaxis, :])[0]
        expected = silu_vectorized(positive_values)
        self.assertAllClose(result, expected, rtol=1e-05)

        # Test values around zero (to ensure sigmoid behaves correctly)
        around_zero_values = np.random.uniform(-1, 1, (2, 5))
        result = activations.silu(around_zero_values[np.newaxis, :])[0]
        expected = silu_vectorized(around_zero_values)
        self.assertAllClose(result, expected, rtol=1e-05)

        # Test negative values
        negative_values = np.random.uniform(-5.9, 0, (2, 5))
        result = activations.silu(negative_values[np.newaxis, :])[0]
        expected = silu_vectorized(negative_values)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_gelu(self):
        def gelu(x, approximate=False):
            if approximate:
                return (
                    0.5
                    * x
                    * (
                        1.0
                        + np.tanh(
                            np.sqrt(2.0 / np.pi)
                            * (x + 0.044715 * np.power(x, 3))
                        )
                    )
                )
            else:
                from scipy.stats import norm

                return x * norm.cdf(x)

        x = np.random.random((2, 5))
        result = activations.gelu(x[np.newaxis, :])[0]
        expected = gelu(x)
        self.assertAllClose(result, expected, rtol=1e-05)

        x = np.random.random((2, 5))
        result = activations.gelu(x[np.newaxis, :], approximate=True)[0]
        expected = gelu(x, True)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_elu(self):
        x = np.random.random((2, 5))
        result = activations.elu(x[np.newaxis, :])[0]
        self.assertAllClose(result, x, rtol=1e-05)
        negative_values = np.array([[-1, -2]], dtype=backend.floatx())
        result = activations.elu(negative_values[np.newaxis, :])[0]
        true_result = np.exp(negative_values) - 1
        self.assertAllClose(result, true_result)

    def test_tanh(self):
        x = np.random.random((2, 5))
        result = activations.tanh(x[np.newaxis, :])[0]
        expected = np.tanh(x)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_exponential(self):
        x = np.random.random((2, 5))
        result = activations.exponential(x[np.newaxis, :])[0]
        expected = np.exp(x)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_mish(self):
        x = np.random.random((2, 5))
        result = activations.mish(x[np.newaxis, :])[0]
        expected = x * np.tanh(_ref_softplus(x))
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_linear(self):
        x = np.random.random((10, 5))
        self.assertAllClose(x, activations.linear(x))

    def test_get_method(self):
        obj = activations.get("relu")
        self.assertEqual(obj, activations.relu)

        obj = activations.get(None)
        self.assertEqual(obj, activations.linear)

        with self.assertRaises(ValueError):
            activations.get("typo")
