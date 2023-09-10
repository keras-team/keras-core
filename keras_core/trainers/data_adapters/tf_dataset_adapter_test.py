import numpy as np
import tensorflow as tf

from keras_core import testing
from keras_core.trainers.data_adapters import tf_dataset_adapter


class TestTFDatasetAdapter(testing.TestCase):
    def test_basic_flow(self):
        x = tf.random.normal((34, 4))
        y = tf.random.normal((34, 2))
        base_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(16)
        adapter = tf_dataset_adapter.TFDatasetAdapter(base_ds)

        self.assertEqual(adapter.num_batches, 3)
        self.assertEqual(adapter.batch_size, None)
        self.assertEqual(adapter.has_partial_batch, None)
        self.assertEqual(adapter.partial_batch_size, None)

        gen = adapter.get_numpy_iterator()
        for i, batch in enumerate(gen):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertIsInstance(bx, np.ndarray)
            self.assertIsInstance(by, np.ndarray)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(bx.dtype, "float32")
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
            self.assertIsInstance(bx, tf.Tensor)
            self.assertIsInstance(by, tf.Tensor)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(bx.dtype, "float32")
            if i < 2:
                self.assertEqual(tuple(bx.shape), (16, 4))
                self.assertEqual(tuple(by.shape), (16, 2))
            else:
                self.assertEqual(tuple(bx.shape), (2, 4))
                self.assertEqual(tuple(by.shape), (2, 2))

    def _test_class_weights(self, target_encoding="int"):
        x = np.random.random((4, 2))
        if target_encoding == "int":
            y = np.array([[0], [1], [2], [3]], dtype="int64")
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
        base_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(16)
        adapter = tf_dataset_adapter.TFDatasetAdapter(
            base_ds, class_weight=class_weight
        )
        gen = adapter.get_numpy_iterator()
        for batch in gen:
            self.assertEqual(len(batch), 3)
            _, _, bw = batch
            self.assertAllClose(bw, [0.1, 0.2, 0.3, 0.4])

    def test_class_weights_int_targets(self):
        self._test_class_weights(target_encoding="int")

    def test_class_weights_categorical_targets(self):
        self._test_class_weights(target_encoding="categorical")

    def test_num_batches(self):
        dataset = tf.data.Dataset.range(42)
        cardinality = int(dataset.cardinality())
        self.assertEqual(cardinality, 42)
        adapter = tf_dataset_adapter.TFDatasetAdapter(dataset)
        self.assertEqual(adapter.num_batches, 42)

        # Test for Infiniate Cardinality
        dataset = tf.data.Dataset.range(42)
        dataset = dataset.repeat()
        cardinality = int(dataset.cardinality())
        self.assertEqual(cardinality, tf.data.INFINITE_CARDINALITY)
        adapter = tf_dataset_adapter.TFDatasetAdapter(dataset)
        self.assertIsNone(adapter.num_batches)

        # Test for Unknown Cardinality
        dataset = dataset.filter(lambda x: True)
        cardinality = int(dataset.cardinality())
        self.assertEqual(cardinality, tf.data.UNKNOWN_CARDINALITY)
        adapter = tf_dataset_adapter.TFDatasetAdapter(dataset)
        self.assertIsNone(adapter.num_batches)

    def test_invalid_dataset_type(self):
        invalid_dataset = [1, 2, 3]
        with self.assertRaisesRegex(
            ValueError, "Expected argument `dataset` to be a tf.data.Dataset."
        ):
            tf_dataset_adapter.TFDatasetAdapter(invalid_dataset)

    def test_sample_weight_with_class_weight(self):
        x = np.random.random((4, 2))
        y = np.array([[0], [1], [2], [3]], dtype="int64")
        sample_weights = np.array([0.1, 0.9, 0.8, 0.2])

        base_ds = tf.data.Dataset.from_tensor_slices(
            (x, y, sample_weights)
        ).batch(16)
        class_weight = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        with self.assertRaisesRegex(
            ValueError,
            "You cannot `class_weight` and `sample_weight` at the same time.",
        ):
            tf_dataset_adapter.TFDatasetAdapter(
                base_ds, class_weight=class_weight
            )

    def test_nested_y_with_class_weight(self):
        x = np.random.random((4, 2))
        y = [
            np.array([[0], [1], [2], [3]], dtype="int64"),
            np.array([[0], [1], [2], [3]], dtype="int64"),
        ]
        base_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(16)
        class_weight = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        with self.assertRaisesRegex(
            ValueError,
            "`class_weight` is only supported for Models with a single output.",
        ):
            tf_dataset_adapter.TFDatasetAdapter(
                base_ds, class_weight=class_weight
            )

    def test_different_y_shapes_with_class_weight(self):
        x = np.random.random((4, 2))
        y = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype="float32",
        )
        base_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(16)
        class_weight = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        adapter = tf_dataset_adapter.TFDatasetAdapter(
            base_ds, class_weight=class_weight
        )
        gen = adapter.get_numpy_iterator()
        for batch in gen:
            _, _, bw = batch
            self.assertAllClose(bw, [0.1, 0.2, 0.3, 0.4])

        y_sparse = np.array([0, 1, 2, 3], dtype="int64")
        base_ds = tf.data.Dataset.from_tensor_slices((x, y_sparse)).batch(16)
        adapter = tf_dataset_adapter.TFDatasetAdapter(
            base_ds, class_weight=class_weight
        )
        gen = adapter.get_numpy_iterator()
        for batch in gen:
            _, _, bw = batch
            self.assertAllClose(bw, [0.1, 0.2, 0.3, 0.4])