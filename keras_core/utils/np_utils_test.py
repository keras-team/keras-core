"""Tests for np_utils."""

import numpy as np
from absl.testing import parameterized

from keras_core import testing
from keras_core.utils import np_utils

NUM_CLASSES = 5


class TestNPUtils(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        [
            {"shape": (1,), "expected_shape": (1, NUM_CLASSES)},
            {"shape": (3,), "expected_shape": (3, NUM_CLASSES)},
            {"shape": (4, 3), "expected_shape": (4, 3, NUM_CLASSES)},
            {"shape": (5, 4, 3), "expected_shape": (5, 4, 3, NUM_CLASSES)},
            {"shape": (3, 1), "expected_shape": (3, NUM_CLASSES)},
            {"shape": (3, 2, 1), "expected_shape": (3, 2, NUM_CLASSES)},
        ]
    )
    def test_to_categorical(self, shape, expected_shape):
        label = np.random.randint(0, NUM_CLASSES, shape)
        one_hot = np_utils.to_categorical(label, NUM_CLASSES)
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
        one_hot = np_utils.to_categorical(label)
        self.assertEqual(one_hot.shape, (3, 5 + 1))

    @parameterized.parameters(
        [
            {"shape": (1,), "expected_shape": (1, NUM_CLASSES - 1)},
            {"shape": (3,), "expected_shape": (3, NUM_CLASSES - 1)},
            {"shape": (4, 3), "expected_shape": (4, 3, NUM_CLASSES - 1)},
            {"shape": (5, 4, 3), "expected_shape": (5, 4, 3, NUM_CLASSES - 1)},
            {"shape": (3, 1), "expected_shape": (3, NUM_CLASSES - 1)},
            {"shape": (3, 2, 1), "expected_shape": (3, 2, NUM_CLASSES - 1)},
        ]
    )
    def test_to_ordinal(self, shape, expected_shape):
        label = np.random.randint(0, NUM_CLASSES, shape)
        ordinal = np_utils.to_ordinal(label, NUM_CLASSES)
        # Check shape
        self.assertEqual(ordinal.shape, expected_shape)
        # Make sure all the values are either 0 or 1
        self.assertTrue(np.all(np.logical_or(ordinal == 0, ordinal == 1)))
        # Get original labels back from ordinal matrix
        self.assertTrue(
            np.all(ordinal.cumprod(-1).sum(-1).reshape(label.shape) == label)
        )

    def test_to_ordinal_without_num_classes(self):
        label = [0, 2, 5]
        one_hot = np_utils.to_ordinal(label)
        self.assertEqual(one_hot.shape, (3, 5))
