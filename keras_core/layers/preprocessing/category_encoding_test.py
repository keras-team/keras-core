import numpy as np
import tensorflow as tf

from keras_core import layers
from keras_core import testing


class CategoryEncodingTest(testing.TestCase):
    # TODO: test coverage is insufficient.
    # Must test > 1 batch size, unbatch cases separately
    def test_count_output(self):
        input_array = np.array([1, 2, 3, 1])
        expected_output = np.array([0, 2, 1, 1, 0, 0])

        num_tokens = 6
        expected_output_shape = (num_tokens,)

        layer = layers.CategoryEncoding(num_tokens=6, output_mode="count")
        int_data = layer(input_array)
        self.assertEqual(expected_output_shape, int_data.shape)
        self.assertAllClose(int_data, expected_output)

    def test_batched_count_output(self):
        input_array = np.array([[1, 2, 3, 1], [0, 3, 1, 0]])
        expected_output = np.array([[0, 2, 1, 1, 0, 0], [2, 1, 0, 1, 0, 0]])

        num_tokens = 6
        expected_output_shape = (2, num_tokens)

        layer = layers.CategoryEncoding(num_tokens=6, output_mode="count")
        int_data = layer(input_array)
        self.assertEqual(expected_output_shape, int_data.shape)
        self.assertAllClose(int_data, expected_output)

    def test_multi_hot(self):
        input_data = np.array([3, 2, 0, 1])
        expected_output = np.array([1, 1, 1, 1, 0, 0])
        num_tokens = 6
        expected_output_shape = (num_tokens,)

        # Test call on layer directly.
        layer = layers.CategoryEncoding(
            num_tokens=num_tokens, output_mode="multi_hot"
        )
        output_data = layer(input_data)
        self.assertAllClose(expected_output, output_data)
        self.assertEqual(expected_output_shape, output_data.shape)

    def test_batched_multi_hot(self):
        input_data = np.array([[3, 2, 0, 1], [3, 2, 0, 1]])
        expected_output = np.array([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]])
        num_tokens = 6
        expected_output_shape = (2, num_tokens)

        # Test call on layer directly.
        layer = layers.CategoryEncoding(
            num_tokens=num_tokens, output_mode="multi_hot"
        )
        output_data = layer(input_data)
        self.assertAllClose(expected_output, output_data)
        self.assertEqual(expected_output_shape, output_data.shape)

    def test_one_hot(self):
        input_data = np.array([3, 2, 0, 1])
        expected_output = np.array(
            [
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ]
        )
        num_tokens = 4
        expected_output_shape = (num_tokens, num_tokens)

        # Test call on layer directly.
        layer = layers.CategoryEncoding(
            num_tokens=num_tokens, output_mode="one_hot"
        )
        output_data = layer(input_data)
        self.assertAllClose(expected_output, output_data)
        self.assertEqual(expected_output_shape, output_data.shape)

    def test_batched_one_hot(self):
        input_data = np.array([[3, 2, 0, 1], [3, 2, 0, 1]])
        expected_output = np.array(
            [
                [
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                ],
                [
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                ],
            ]
        )
        num_tokens = 4
        expected_output_shape = (2, num_tokens, num_tokens)

        # Test call on layer directly.
        layer = layers.CategoryEncoding(
            num_tokens=num_tokens, output_mode="one_hot"
        )
        output_data = layer(input_data)
        self.assertAllClose(expected_output, output_data)
        self.assertEqual(expected_output_shape, output_data.shape)

    def test_tf_data_compatibility(self):
        layer = layers.CategoryEncoding(
            num_tokens=4, output_mode="one_hot", dtype="int32"
        )
        input_data = np.array([3, 2, 0, 1]).astype("int64")
        expected_output = np.array(
            [
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ]
        )
        ds = tf.data.Dataset.from_tensor_slices(input_data).batch(4).map(layer)
        for output in ds.take(1):
            output = output.numpy()
        self.assertAllClose(output, expected_output)
