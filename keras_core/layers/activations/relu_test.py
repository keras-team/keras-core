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
                "max_value" : 10,
                "negative_slope" : 1,
                "threshold" : 0.5
            },
            input_shape=(2, 3, 4),
            supports_masking=True,
        )
