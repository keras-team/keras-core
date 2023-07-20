import jax
import numpy as np
import pandas
import tensorflow as tf
import torch
from absl.testing import parameterized

from keras_core import backend
from keras_core import testing
from keras_core.trainers.data_adapters import array_data_adapter


class TestArrayDataAdapter(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        [("np",), ("tf",), ("jax",), ("torch",), ("pandas")]
    )
    def test_basic_flow(self, array_type):
        if array_type == "np":
            x = np.random.random((34, 4))
            y = np.random.random((34, 2))
        elif array_type == "tf":
            x = tf.random.normal((34, 4))
            y = tf.random.normal((34, 2))
        elif array_type == "jax":
            x = jax.numpy.ones((34, 4))
            y = jax.numpy.ones((34, 2))
        elif array_type == "torch":
            x = torch.ones((34, 4))
            y = torch.ones((34, 2))
        elif array_type == "pandas":
            x = pandas.DataFrame(np.random.random((34, 4)))
            y = pandas.DataFrame(np.random.random((34, 2)))

        adapter = array_data_adapter.ArrayDataAdapter(
            x,
            y=y,
            sample_weight=None,
            batch_size=16,
            steps=None,
            shuffle=False,
        )
        self.assertEqual(adapter.num_batches, 3)
        self.assertEqual(adapter.batch_size, 16)
        self.assertEqual(adapter.has_partial_batch, True)
        self.assertEqual(adapter.partial_batch_size, 2)

        gen = adapter.get_numpy_iterator()
        for i, batch in enumerate(gen):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertTrue(isinstance(bx, np.ndarray))
            self.assertTrue(isinstance(by, np.ndarray))
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(bx.dtype, backend.floatx())
            if i < 2:
                self.assertEqual(bx.shape, (16, 4))
                self.assertEqual(by.shape, (16, 2))
            else:
                self.assertEqual(bx.shape, (2, 4))
                self.assertEqual(by.shape, (2, 2))
        ds = adapter.get_tf_dataset()
        for i, batch in enumerate(ds):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertTrue(isinstance(bx, tf.Tensor))
            self.assertTrue(isinstance(by, tf.Tensor))
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(bx.dtype, backend.floatx())
            if i < 2:
                self.assertEqual(tuple(bx.shape), (16, 4))
                self.assertEqual(tuple(by.shape), (16, 2))
            else:
                self.assertEqual(tuple(bx.shape), (2, 4))
                self.assertEqual(tuple(by.shape), (2, 2))

    def test_multi_inputs_and_outputs(self):
        x1 = np.random.random((34, 1))
        x2 = np.random.random((34, 2))
        y1 = np.random.random((34, 3))
        y2 = np.random.random((34, 4))
        sw = np.random.random((34,))
        adapter = array_data_adapter.ArrayDataAdapter(
            x={"x1": x1, "x2": x2},
            y=[y1, y2],
            sample_weight=sw,
            batch_size=16,
            steps=None,
            shuffle=False,
        )
        gen = adapter.get_numpy_iterator()
        for i, batch in enumerate(gen):
            self.assertEqual(len(batch), 3)
            bx, by, bw = batch
            self.assertTrue(isinstance(bx, dict))
            # NOTE: the y list was converted to a tuple for tf.data
            # compatibility.
            self.assertTrue(isinstance(by, tuple))
            self.assertTrue(isinstance(bw, tuple))

            self.assertTrue(isinstance(bx["x1"], np.ndarray))
            self.assertTrue(isinstance(bx["x2"], np.ndarray))
            self.assertTrue(isinstance(by[0], np.ndarray))
            self.assertTrue(isinstance(by[1], np.ndarray))
            self.assertTrue(isinstance(bw[0], np.ndarray))
            self.assertTrue(isinstance(bw[1], np.ndarray))

            self.assertEqual(bx["x1"].dtype, by[0].dtype)
            self.assertEqual(bx["x1"].dtype, backend.floatx())
            if i < 2:
                self.assertEqual(bx["x1"].shape, (16, 1))
                self.assertEqual(bx["x2"].shape, (16, 2))
                self.assertEqual(by[0].shape, (16, 3))
                self.assertEqual(by[1].shape, (16, 4))
                self.assertEqual(bw[0].shape, (16,))
                self.assertEqual(bw[1].shape, (16,))
            else:
                self.assertEqual(bx["x1"].shape, (2, 1))
                self.assertEqual(by[0].shape, (2, 3))
                self.assertEqual(bw[0].shape, (2,))
                self.assertEqual(bw[1].shape, (2,))
        ds = adapter.get_tf_dataset()
        for i, batch in enumerate(ds):
            self.assertEqual(len(batch), 3)
            bx, by, bw = batch
            self.assertTrue(isinstance(bx, dict))
            # NOTE: the y list was converted to a tuple for tf.data
            # compatibility.
            self.assertTrue(isinstance(by, tuple))
            self.assertTrue(isinstance(bw, tuple))

            self.assertTrue(isinstance(bx["x1"], tf.Tensor))
            self.assertTrue(isinstance(bx["x2"], tf.Tensor))
            self.assertTrue(isinstance(by[0], tf.Tensor))
            self.assertTrue(isinstance(by[1], tf.Tensor))
            self.assertTrue(isinstance(bw[0], tf.Tensor))
            self.assertTrue(isinstance(bw[1], tf.Tensor))

            self.assertEqual(bx["x1"].dtype, by[0].dtype)
            self.assertEqual(bx["x1"].dtype, backend.floatx())
            if i < 2:
                self.assertEqual(tuple(bx["x1"].shape), (16, 1))
                self.assertEqual(tuple(bx["x2"].shape), (16, 2))
                self.assertEqual(tuple(by[0].shape), (16, 3))
                self.assertEqual(tuple(by[1].shape), (16, 4))
                self.assertEqual(tuple(bw[0].shape), (16,))
                self.assertEqual(tuple(bw[1].shape), (16,))
            else:
                self.assertEqual(tuple(bx["x1"].shape), (2, 1))
                self.assertEqual(tuple(by[0].shape), (2, 3))
                self.assertEqual(tuple(bw[0].shape), (2,))
                self.assertEqual(tuple(bw[1].shape), (2,))

    @parameterized.parameters([("int",), ("categorical",)])
    def test_class_weights(self, target_encoding):
        x = np.random.random((4, 2))
        if target_encoding == "int":
            y = np.array([[0], [1], [2], [3]], dtype="int32")
        else:
            y = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype="float32",
            )

        class_weight = {
            0: 0.1,
            1: 0.2,
            2: 0.3,
            3: 0.4,
        }
        adapter = array_data_adapter.ArrayDataAdapter(
            x,
            y=y,
            class_weight=class_weight,
            batch_size=16,
        )
        gen = adapter.get_numpy_iterator()
        for batch in gen:
            self.assertEqual(len(batch), 3)
            _, _, bw = batch
            self.assertAllClose(bw, [0.1, 0.2, 0.3, 0.4])

    def test_errors(self):
        # TODO
        pass
