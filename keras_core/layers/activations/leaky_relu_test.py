import numpy as np

from keras_core import testing
from keras_core.layers.activations import leaky_relu


class ReLUTest(testing.TestCase):
    def test_relu(self):
        self.run_layer_test(
            leaky_relu.LeakyReLU,
            init_kwargs={
                "negative_slope": 1,
            },
            input_shape=(2, 3, 4),
            supports_masking=True,
        )

    def test_leaky_relu_correctness(self):
        leaky_relu_layer = leaky_relu.LeakyReLU(negative_slope=0.5)
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
                leaky_relu.LeakyReLU,
                init_kwargs={"negative_slope": None},
                input_shape=(2, 3, 4),
                supports_masking=True,
            )
