import jax
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from jax import numpy as jnp

from keras_core import backend
from keras_core import operations as ops
from keras_core import testing
from keras_core.utils import numerical_utils

NUM_CLASSES = 5


class TestNumericalUtils(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        [
            ((1,), (1, NUM_CLASSES)),
            ((3,), (3, NUM_CLASSES)),
            ((4, 3), (4, 3, NUM_CLASSES)),
            ((5, 4, 3), (5, 4, 3, NUM_CLASSES)),
            ((3, 1), (3, NUM_CLASSES)),
            ((3, 2, 1), (3, 2, NUM_CLASSES)),
        ]
    )
    def test_to_categorical(self, shape, expected_shape):
        label = np.random.randint(0, NUM_CLASSES, shape)
        one_hot = numerical_utils.to_categorical(label, NUM_CLASSES)
        # Check shape
        self.assertEqual(one_hot.shape, expected_shape)
        # Make sure there is only one 1 in a row
        self.assertTrue(np.all(one_hot.sum(axis=-1) == 1))
        # Get original labels back from one hots
        self.assertTrue(
            np.all(np.argmax(one_hot, -1).reshape(label.shape) == label)
        )

    def test_to_categorial_without_num_classes(self):
        label = [0, 2, 5]
        one_hot = numerical_utils.to_categorical(label)
        self.assertEqual(one_hot.shape, (3, 5 + 1))

    def test_to_categorical_with_backend_tensor(self):
        label = backend.convert_to_tensor(np.array([0, 2, 1, 3, 4]))
        expected = backend.convert_to_tensor(
            np.array(
                [
                    [1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                ]
            )
        )
        one_hot = numerical_utils.to_categorical(label, NUM_CLASSES)
        assert backend.is_tensor(one_hot)
        assert all(ops.equal(one_hot, expected))
