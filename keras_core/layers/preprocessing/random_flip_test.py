import unittest.mock

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from keras_core import backend
from keras_core import layers
from keras_core import testing
from keras_core import utils


class MockedRandomFlip(layers.RandomFlip):
    def call(self, inputs, training=True):
        with unittest.mock.patch.object(
            self.backend.random, "uniform", return_value=0.1
        ):
            out = super().call(inputs, training=training)
        return out


class RandomFlipTest(testing.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("random_flip_horizontal", "horizontal"),
        ("random_flip_vertical", "vertical"),
        ("random_flip_both", "horizontal_and_vertical"),
    )
    def test_random_flip(self, mode):
        self.run_layer_test(
            layers.RandomFlip,
            init_kwargs={
                "mode": mode,
            },
            input_shape=(2, 3, 4),
            expected_output_shape=(2, 3, 4),
            supports_masking=False,
            run_training_check=False,
        )

    def test_random_flip_horizontal(self):
        utils.set_random_seed(0)
        self.run_layer_test(
            MockedRandomFlip,
            init_kwargs={
                "mode": "horizontal",
                "seed": 42,
            },
            input_data=np.asarray([[[2, 3, 4], [5, 6, 7]]]),
            expected_output=backend.convert_to_tensor([[[5, 6, 7], [2, 3, 4]]]),
            supports_masking=False,
            run_training_check=False,
        )

    def test_random_flip_vertical(self):
        utils.set_random_seed(0)
        self.run_layer_test(
            MockedRandomFlip,
            init_kwargs={
                "mode": "vertical",
                "seed": 42,
            },
            input_data=np.asarray([[[2, 3, 4]], [[5, 6, 7]]]),
            expected_output=backend.convert_to_tensor(
                [[[5, 6, 7]], [[2, 3, 4]]]
            ),
            supports_masking=False,
            run_training_check=False,
        )

    def test_support_jit(self):
        if backend.backend() not in ("tensorflow", "jax"):
            self.skipTest(
                "only tensorflow and jax support jit compilation for RandomFlip"
            )
        input_data = np.random.random(size=(2, 4, 4, 3)) * 255
        layer = layers.RandomFlip()
        if backend.backend() == "jax":
            import jax

            @jax.jit
            def call(input_data):
                return layer.call(input_data)

        elif backend.backend() == "tensorflow":

            @tf.function(jit_compile=True)
            def call(input_data):
                return layer.call(input_data)

        call(input_data)

    def test_tf_data_compatibility(self):
        layer = layers.RandomFlip("vertical", seed=42)
        input_data = np.array([[[2, 3, 4]], [[5, 6, 7]]])
        expected_output = np.array([[[5, 6, 7]], [[2, 3, 4]]])
        ds = tf.data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output = output.numpy()
        self.assertAllClose(output, expected_output)
