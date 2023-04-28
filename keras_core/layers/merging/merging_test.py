import numpy as np

from keras_core import backend
from keras_core import layers
from keras_core import models
from keras_core import testing


class MergingLayersTest(testing.TestCase):
    def test_add(self):
        input_1 = layers.Input(shape=(4, 5))
        input_2 = layers.Input(shape=(4, 5))
        add_layer = layers.Add()
        out = add_layer([input_1, input_2])
        model = models.Model([input_1, input_2], out)

        self.assertListEqual(list(out.shape), [None, 4, 5])

        x1 = np.random.rand(*(2, 4, 5))
        x2 = np.random.rand(*(2, 4, 5))
        out = model([x1, x2])
        self.assertEqual(out.shape, (2, 4, 5))
        self.assertAllClose(out, x1 + x2, atol=1e-4)

        self.assertIsNone(
            add_layer.compute_mask([input_1, input_2], [None, None])
        )
        self.assertTrue(
            np.all(
                add_layer.compute_mask(
                    [input_1, input_2],
                    [backend.Variable(x1), backend.Variable(x2)],
                )
            )
        )
        with self.assertRaisesRegex(ValueError, "`mask` should be a list."):
            add_layer.compute_mask([input_1, input_2], x1)

        with self.assertRaisesRegex(ValueError, "`inputs` should be a list."):
            add_layer.compute_mask(input_1, [None, None])

        with self.assertRaisesRegex(
            ValueError, " should have the same length."
        ):
            add_layer.compute_mask([input_1, input_2], [None])
