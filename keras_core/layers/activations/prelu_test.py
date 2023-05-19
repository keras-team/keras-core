import numpy as np

from keras_core import testing
from keras_core.layers.activations import prelu
import tensorflow as tf


class PReLUTest(testing.TestCase):
    def test_prelu(self):
        self.run_layer_test(
            prelu.PReLU,
            init_kwargs={
                "negative_slope_initializer": "zeros",
                "negative_slope_regularizer": "L1",
                "negative_slope_constraint": "MaxNorm",
                "shared_axes": 1,
            },
            input_shape=(2, 3, 4),
            supports_masking=True,
        )

    def test_prelu_correctness(self):
        inputs = np.random.randn(2, 10, 5, 3)
        prelu_layer = prelu.PReLU(
            negative_slope_initializer="glorot_uniform",
            negative_slope_regularizer="l1",
            negative_slope_constraint="non_neg",
            shared_axes=(1, 2),
        )
        tf_prelu_layer = tf.keras.layers.PReLU(
            negative_slope_initializer="glorot_uniform",
            negative_slope_regularizer="l1",
            negative_slope_constraint="non_neg",
            shared_axes=(1, 2),
        )

        prelu_layer.build(inputs.shape)
        tf_prelu_layer.build(inputs.shape)

        weights = np.random.random(inputs.shape[3:])
        prelu_layer.negative_slope.assign(weights)
        tf_prelu_layer.negative_slope.assign(weights)

        self.assertAllClose(prelu_layer(inputs), tf_prelu_layer(inputs))
