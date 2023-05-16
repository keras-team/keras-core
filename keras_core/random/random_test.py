from absl.testing import parameterized

import numpy as np

from keras_core.random import random
from keras_core import testing
from keras_core.backend.common.keras_tensor import KerasTensor
import keras_core.operations as ops


class RandomTest(testing.TestCase, parameterized.TestCase):

    @parameterized.parameters(
            {"seed": 10, "shape": (5,), "mean": 0, "stddev": 1},
            {"seed": 10, "shape": (2, 3), "mean": 0, "stddev": 1},
            {"seed": 10, "shape": (2, 3, 4), "mean": 0, "stddev": 1},
            {"seed": 10, "shape": (2, 3), "mean": 10, "stddev": 1},
            {"seed": 10, "shape": (2, 3), "mean": 10, "stddev": 3},
    )
    def test_normal(self, seed, shape, mean, stddev):
        np.random.seed(seed)
        np_res = np.random.normal(loc=mean, scale=stddev, size=shape)
        res = random.normal(shape, mean=mean, stddev=stddev, seed=seed)
        self.assertEqual(res.shape, KerasTensor(shape).shape)
        self.assertEqual(res.shape, np_res.shape)

    @parameterized.parameters(
            {"seed": 10, "shape": (5,), "minval": 0, "maxval": 1},
            {"seed": 10, "shape": (2, 3), "minval": 0, "maxval": 1},
            {"seed": 10, "shape": (2, 3, 4), "minval": 0, "maxval": 2},
            {"seed": 10, "shape": (2, 3), "minval": -1, "maxval": 1},
            {"seed": 10, "shape": (2, 3), "minval": 1, "maxval": 3},
    )
    def test_uniform(self, seed, shape, minval, maxval):
        np.random.seed(seed)
        np_res = np.random.uniform(low=minval, high=maxval, size=shape)
        res = random.uniform(shape, minval=minval, maxval=maxval, seed=seed)
        self.assertEqual(res.shape, KerasTensor(shape).shape)
        self.assertEqual(res.shape, np_res.shape)
        self.assertLessEqual(ops.max(res), maxval)
        self.assertGreaterEqual(ops.max(res), minval)

    @parameterized.parameters(
            {"seed": 10, "shape": (5,), "mean": 0, "stddev": 1},
            {"seed": 10, "shape": (2, 3), "mean": 0, "stddev": 1},
            {"seed": 10, "shape": (2, 3, 4), "mean": 0, "stddev": 1},
            {"seed": 10, "shape": (2, 3), "mean": 10, "stddev": 1},
            {"seed": 10, "shape": (2, 3), "mean": 10, "stddev": 3},
    )
    def test_truncated_normal(self, seed, shape, mean, stddev):
        np.random.seed(seed)
        np_res = np.random.normal(loc=mean, scale=stddev, size=shape)
        res = random.truncated_normal(shape, mean=mean, stddev=stddev, seed=seed)
        self.assertEqual(res.shape, KerasTensor(shape).shape)
        self.assertEqual(res.shape, np_res.shape)
        self.assertLessEqual(ops.max(res), mean + 2 * stddev)
        self.assertGreaterEqual(ops.max(res), mean - 2 * stddev)


    def test_dropout(self):
        # TODO: add dropout test
        pass