import numpy as np
import tensorflow as tf

from keras_core import layers
from keras_core import testing


class HashedCrossingTest(testing.TestCase):
    def test_basics(self):
        self.run_layer_test(
            layers.HashedCrossing,
            init_kwargs={
                "num_bins": 3,
                "output_mode": "int",
            },
            input_data=([1, 2], [4, 5]),
            expected_output_shape=(2,),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
            run_training_check=False,
        )
        self.run_layer_test(
            layers.HashedCrossing,
            init_kwargs={"num_bins": 4, "output_mode": "one_hot"},
            input_data=([1, 2], [4, 5]),
            expected_output_shape=(2, 4),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            expected_num_seed_generators=0,
            expected_num_losses=0,
            supports_masking=False,
            run_training_check=False,
        )

    def test_correctness(self):
        layer = layers.HashedCrossing(num_bins=10)
        feat1 = tf.constant("A")
        feat2 = tf.constant(101)
        outputs = layer((feat1, feat2))
        self.assertAllClose(outputs, 1)
        self.assertAllEqual(outputs.shape.as_list(), [])

        layer = layers.HashedCrossing(num_bins=5)
        feat1 = np.array(["A", "B", "A", "B", "A"])
        feat2 = np.array([101, 101, 101, 102, 102])
        output = layer((feat1, feat2))
        self.assertAllClose(np.array([1, 4, 1, 1, 3]), output)

        layer = layers.HashedCrossing(num_bins=5, output_mode="one_hot")
        feat1 = np.array(["A", "B", "A", "B", "A"])
        feat2 = np.array([101, 101, 101, 102, 102])
        output = layer((feat1, feat2))
        self.assertAllClose(
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                ]
            ),
            output,
        )

    def test_tf_data_compatibility(self):
        layer = layers.HashedCrossing(num_bins=5)
        feat1 = np.array(["A", "B", "A", "B", "A"])
        feat2 = np.array([101, 101, 101, 102, 102])
        ds = (
            tf.data.Dataset.from_tensor_slices((feat1, feat2))
            .batch(5)
            .map(lambda x1, x2: layer((x1, x2)))
        )
        for output in ds.take(1):
            output = output.numpy()
        self.assertAllClose(np.array([1, 4, 1, 1, 3]), output)

    def test_upsupported_shape_input_fails(self):
        with self.assertRaisesRegex(ValueError, "inputs should have shape"):
            layers.HashedCrossing(num_bins=10)(
                (tf.constant([[[1.0]]]), tf.constant([[[1.0]]]))
            )

    def test_cross_output_dtype(self):
        layer = layers.HashedCrossing(num_bins=2)
        self.assertEqual(layer(([1], [1])).dtype, tf.int64)
        layer = layers.HashedCrossing(num_bins=2, dtype=tf.int32)
        self.assertEqual(layer(([1], [1])).dtype, tf.int32)
        layer = layers.HashedCrossing(num_bins=2, output_mode="one_hot")
        self.assertEqual(layer(([1], [1])).dtype, tf.float32)
        layer = layers.HashedCrossing(
            num_bins=2, output_mode="one_hot", dtype=tf.float64
        )
        self.assertEqual(layer(([1], [1])).dtype, tf.float64)

    def test_non_list_input_fails(self):
        with self.assertRaisesRegex(ValueError, "should be called on a list"):
            layers.HashedCrossing(num_bins=10)(tf.constant(1))

    def test_single_input_fails(self):
        with self.assertRaisesRegex(ValueError, "at least two inputs"):
            layers.HashedCrossing(num_bins=10)([tf.constant(1)])

    def test_sparse_input_fails(self):
        with self.assertRaisesRegex(
            ValueError, "inputs should be dense tensors"
        ):
            sparse_in = tf.sparse.from_dense(tf.constant([1]))
            layers.HashedCrossing(num_bins=10)((sparse_in, sparse_in))

    def test_float_input_fails(self):
        with self.assertRaisesRegex(
            ValueError, "should have an integer or string"
        ):
            layers.HashedCrossing(num_bins=10)(
                (tf.constant([1.0]), tf.constant([1.0]))
            )

    def test_from_config(self):
        layer = layers.HashedCrossing(
            num_bins=5, output_mode="one_hot", sparse=True
        )
        cloned_layer = layers.HashedCrossing.from_config(layer.get_config())
        feat1 = tf.constant([["A"], ["B"], ["A"], ["B"], ["A"]])
        feat2 = tf.constant([[101], [101], [101], [102], [102]])
        original_outputs = layer((feat1, feat2))
        cloned_outputs = cloned_layer((feat1, feat2))
        self.assertAllClose(
            tf.sparse.to_dense(cloned_outputs),
            tf.sparse.to_dense(original_outputs),
        )
