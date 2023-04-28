import numpy as np
import pytest

from keras_core import backend
from keras_core import layers
from keras_core import models
from keras_core import operations as ops
from keras_core import testing


@pytest.mark.skipif(
    backend() != "tensorflow",
    reason="Dynamic shapes are only supported in TensorFlow backend.",
)
class MergingLayersDynamicShapeTest(testing.TestCase):
    def test_add(self):
        batch_size = 2
        shape = (4, 5)
        x1 = np.random.rand(batch_size, *shape)
        x2 = np.random.rand(batch_size, *shape)
        x3 = ops.convert_to_tensor(x1 + x2)

        self.run_layer_test(
            layers.Add,
            init_kwargs={},
            input_shape=None,
            input_data=[x1, x2],
            expected_output=x3,
            expected_num_trainable_weights=None,
            expected_num_non_trainable_weights=None,
            expected_num_seed_generators=None,
            expected_num_losses=None,
            supports_masking=True,
        )

        input_1 = layers.Input(shape=shape, batch_size=batch_size)
        input_2 = layers.Input(shape=shape, batch_size=batch_size)
        add_layer = layers.Add()
        out = add_layer([input_1, input_2])
        model = models.Model([input_1, input_2], out)
        res = model([x1, x2])

        self.assertEqual(res.shape, (batch_size, *shape))
        self.assertAllClose(res, x3, atol=1e-4)
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


class MergingLayersStaticShapeTest(testing.TestCase):
    def test_add(self):
        batch_size = 2
        shape = (4, 5)
        x1 = np.random.rand(batch_size, *shape)
        x2 = np.random.rand(batch_size, *shape)
        x3 = ops.convert_to_tensor(x1 + x2)

        self.run_layer_test(
            layers.Add,
            init_kwargs={},
            input_shape=None,
            input_data=[x1, x2],
            expected_output=x3,
            expected_num_trainable_weights=None,
            expected_num_non_trainable_weights=None,
            expected_num_seed_generators=None,
            expected_num_losses=None,
            supports_masking=True,
        )

        input_1 = layers.Input(shape=shape, batch_size=batch_size)
        input_2 = layers.Input(shape=shape, batch_size=batch_size)
        add_layer = layers.Add()
        out = add_layer([input_1, input_2])
        model = models.Model([input_1, input_2], out)
        res = model([x1, x2])

        self.assertEqual(res.shape, (batch_size, *shape))
        self.assertAllClose(res, x3, atol=1e-4)
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
