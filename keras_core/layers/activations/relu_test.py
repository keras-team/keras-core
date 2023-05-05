import numpy as np

from keras_core import testing
from keras_core.layers.activations import relu


class ReLUTest(testing.TestCase):
    def test_config(self):
        relu_layer = relu.ReLU()
        self.run_class_serialization_test(relu_layer)

    def test_relu(self):
        self.run_layer_test(
            relu.ReLU,
            init_kwargs={},
            input_shape=(2, 3, 4),
            supports_masking=True,
        )

        x = np.random.random((2, 5))
        relu_layer = relu.ELU()
        result = relu_layer(x[np.newaxis, :])[0]
        self.assertAllClose(result, x, rtol=1e-05)