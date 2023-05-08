import numpy as np

from keras_core import testing
from keras_core.layers.activations import prelu


class LeakyReLUTest(testing.TestCase):
    def test_relu(self):
        self.run_layer_test(
            prelu.PReLU,
            init_kwargs={
                "negative_slope_initializer": "zeros",
                "negative_slope_regularizer": "L1",
                "negative_slope_constraint": "MaxNorm",
                "shared_axes": None,
            },
            input_shape=(2, 3, 4),
            supports_masking=True,
        )

    def test_leaky_relu_correctness(self):
        leaky_relu_layer = prelu.PReLU(negative_slope=0.5)
        input = np.array([-10, -5, 0.0, 5, 10])
        expected_output = np.array([-5.0, -2.5, 0.0, 5.0, 10.0])
        result = leaky_relu_layer(input)
        self.assertAllClose(result, expected_output)

    def test_invalid_usage(self):
        with self.assertRaisesRegex(
            ValueError,
            "The negative_slope value of a Leaky ReLU layer cannot be None, "
            "Expecting a float. Received negative_slope: None",
        ):
            self.run_layer_test(
                prelu.PReLU,
                init_kwargs={},
                input_shape=(2, 3, 4),
                supports_masking=True,
            )
