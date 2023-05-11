import numpy as np

from keras_core import testing
from keras_core.operations import core as kcore


class CoreOpsCorrectnessTest(testing.TestCase):
    def test_scatter_1d(self):
        indices = np.array([[1], [3], [4], [7]])
        updates = np.array([9, 10, 11, 12])
        self.assertAllClose(
            kcore.scatter(indices, updates, np.array([8])),
            [0, 9, 0, 10, 11, 0, 0, 12],
        )

    def test_scatter_2d(self):
        indices = np.array([[0, 1], [2, 0]])
        updates = np.array([5, 10])
        self.assertAllClose(
            kcore.scatter(indices, updates, (3, 2)), [[0, 5], [0, 0], [10, 0]]
        )

    def test_scatter_slice(self):
        indices = np.array([[2], [4]])
        updates = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertAllClose(
            kcore.scatter(indices, updates, (6, 3)),
            [[0, 0, 0], [0, 0, 0], [1, 2, 3], [0, 0, 0], [4, 5, 6], [0, 0, 0]],
        )
