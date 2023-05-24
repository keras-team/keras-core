import numpy as np

from keras_core import layers
from keras_core import testing
import tensorflow as tf
from absl.testing import parameterized


class BatchNormalizationTest(testing.TestCase, parameterized.TestCase):
    def test_bn_basics(self):
        # vector case
        self.run_layer_test(
            layers.BatchNormalization,
            init_kwargs={
                "center": True,
                "scale": True,
            },
            call_kwargs={"training": True},
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=2,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.BatchNormalization,
            init_kwargs={
                "center": False,
                "scale": False,
            },
            call_kwargs={"training": True},
            input_shape=(2, 3),
            expected_output_shape=(2, 3),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=2,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=True,
        )
        # image case, with regularizers
        self.run_layer_test(
            layers.BatchNormalization,
            init_kwargs={
                "center": True,
                "scale": True,
                "beta_regularizer": "l2",
                "gamma_regularizer": "l2",
            },
            call_kwargs={"training": True},
            input_shape=(2, 4, 4, 3),
            expected_output_shape=(2, 4, 4, 3),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=2,
            expected_num_seed_generators=0,
            expected_num_losses=2,  # we have 2 regularizers.
            supports_masking=True,
        )

    @parameterized.product(
        axis=(-1,),
        input_shape=((1, 2, 2, 1),),
    )
    def test_correctness(self, axis, input_shape):
        # Training
        layer = layers.BatchNormalization(axis=axis, momentum=0)
        tf_keras_layer = tf.keras.layers.BatchNormalization(
            axis=axis, momentum=0
        )
        layer.build(input_shape)
        tf_keras_layer.build(input_shape)
        # Random data centered on 5.0, variance 10.0
        x = np.random.normal(loc=5.0, scale=10.0, size=input_shape)
        x = np.reshape(np.array([[1, 2], [3, 4]]), input_shape)
        out = x
        tf_out = x
        for _ in range(1):
            out = layer(out, training=True)
            tf_out = tf_keras_layer(tf_out, training=True)

        import pdb

        pdb.set_trace()
        self.assertAllClose(out, tf_out)

        # Assert the normalization is correct.
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[axis] = input_shape[axis]
        out -= np.reshape(np.array(layer.beta), broadcast_shape)
        out /= np.reshape(np.array(layer.gamma), broadcast_shape)

        self.assertAllClose(np.mean(out, axis=(0, 1, 2)), 0.0, atol=1e-3)
        self.assertAllClose(np.std(out, axis=(0, 1, 2)), 1.0, atol=1e-3)

        # Inference
        out = layer(x, training=False)
        tf_out = tf_keras_layer(x, training=False)
        self.assertAllClose(out, tf_out, atol=5e-3)

    def test_trainable_behavior(self):
        layer = layers.BatchNormalization(axis=-1, momentum=0.8, epsilon=1e-7)
        layer.build((1, 4, 4, 3))
        layer.trainable = False
        self.assertEqual(len(layer.weights), 4)
        self.assertEqual(len(layer.trainable_weights), 0)
        self.assertEqual(len(layer.non_trainable_weights), 4)

        # Random data centered on 5.0, variance 10.0
        x = np.random.normal(loc=5.0, scale=10.0, size=(200, 4, 4, 3))

        out = layer(x, training=True)
        self.assertAllClose(out, x)

        layer.trainable = True
        self.assertEqual(len(layer.weights), 4)
        self.assertEqual(len(layer.trainable_weights), 2)
        self.assertEqual(len(layer.non_trainable_weights), 2)

        for _ in range(10):
            out = layer(x, training=True)

        out -= np.reshape(np.array(layer.beta), (1, 1, 1, 3))
        out /= np.reshape(np.array(layer.gamma), (1, 1, 1, 3))

        self.assertAllClose(np.mean(out, axis=(0, 1, 2)), 0.0, atol=1e-3)
        self.assertAllClose(np.std(out, axis=(0, 1, 2)), 1.0, atol=1e-3)
