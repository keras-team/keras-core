import numpy as np
from absl.testing import parameterized

from keras_core import ops
from keras_core import testing


class RandomTest(testing.TestCase, parameterized.TestCase):
    def test_shuffle(self):
        x = np.arange(100).reshape(10, 10)

        # Test axis=0
        expected = ops.random.shuffle(x)

        self.assertFalse(
            np.all(x == ops.convert_to_numpy(expected))
        )  # seems unlikely!
        self.assertAllClose(np.sum(x, axis=0), ops.sum(expected, axis=0))
        self.assertNotAllClose(np.sum(x, axis=1), ops.sum(expected, axis=1))

        # Test axis=1
        expected = ops.random.shuffle(x, axis=1)

        self.assertFalse(
            np.all(x == ops.convert_to_numpy(expected))
        )  # seems unlikely!
        self.assertAllClose(np.sum(x, axis=1), ops.sum(expected, axis=1))
        self.assertNotAllClose(np.sum(x, axis=0), ops.sum(expected, axis=0))
