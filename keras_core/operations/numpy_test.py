import numpy as np

from keras_core import backend
from keras_core import testing
from keras_core.backend.keras_tensor import KerasTensor
from keras_core.operations import numpy as knp
from keras_core.operations import operation


class NumpySingleInputOpsShapeTest(testing.TestCase):
    def test_add(self):
        x = KerasTensor((2, 3))
        y = KerasTensor((2, 3))
        self.assertEqual(knp.add(x, y).shape, (2, 3))

        x = KerasTensor((None, 3))
        y = KerasTensor((2, None))
        self.assertEqual(knp.add(x, y).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.add(x, y)

    def test_subtract(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.subtract(x, y).shape, (2, 3))

        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.subtract(x, y).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.subtract(x, y)

    def test_multiply(self):
        x = KerasTensor([2, 3])
        y = KerasTensor([2, 3])
        self.assertEqual(knp.multiply(x, y).shape, (2, 3))

        x = KerasTensor([None, 3])
        y = KerasTensor([2, None])
        self.assertEqual(knp.multiply(x, y).shape, (2, 3))

        with self.assertRaises(ValueError):
            x = KerasTensor([2, 3])
            y = KerasTensor([2, 3, 4])
            knp.multiply(x, y)

    # TODO: add more unit test!
