import numpy as np

from keras_core import testing
from keras_core.layers.activations import relu


class ReLUTest(testing.TestCase):
    def test_config(self):
        relu_layer = relu.ReLU(max_value=10, negative_slope=1, threshold=1)
        self.run_class_serialization_test(relu_layer)

    def test_relu(self):
        self.run_layer_test(
            relu.ReLU,
            init_kwargs={
                "max_value": 10,
                "negative_slope": 1,
                "threshold": 0.5,
            },
            input_shape=(2, 3, 4),
            supports_masking=True,
        )

    def test_relu_correctness(self):
        relu_layer = relu.ReLU(max_value=10, negative_slope=0.5, threshold=0)
        input = np.array([-10, -5, 0.0, 5, 10])
        expected_output = np.array([-5.0, -2.5, 0.0, 5.0, 10.0])
        result = relu_layer(input)
        self.assertAllClose(result, expected_output)

    def test_invalid_usage(self):
        with self.assertRaisesRegex(
            ValueError,
            "max_value of a ReLU layer cannot be a negative "
            "value. Received: -10",
        ):
            self.run_layer_test(
                relu.ReLU,
                init_kwargs={
                    "max_value": -10,
                    "negative_slope": 1,
                    "threshold": 0.5,
                },
                input_shape=(2, 3, 4),
                supports_masking=True,
            )

        with self.assertRaisesRegex(
            ValueError,
            "negative_slope of a ReLU layer cannot be a negative "
            "value. Received: -10",
        ):
            self.run_layer_test(
                relu.ReLU,
                init_kwargs={
                    "max_value": 10,
                    "negative_slope": -10,
                    "threshold": 0.5,
                },
                input_shape=(2, 3, 4),
                supports_masking=True,
            )

        with self.assertRaisesRegex(
            ValueError,
            "threshold of a ReLU layer cannot be a negative "
            "value. Received: -10",
        ):
            self.run_layer_test(
                relu.ReLU,
                init_kwargs={
                    "max_value": 10,
                    "negative_slope": 1,
                    "threshold": -10,
                },
                input_shape=(2, 3, 4),
                supports_masking=True,
            )
