import numpy as np
import tensorflow as tf

from keras_core import layers
from keras_core import testing
from keras_core.models import Model


class CategoryEncodingTest(testing.TestCase):
    def test_dense_oov_input(self):
        valid_array = tf.constant([[0, 1, 2], [0, 1, 2]])
        invalid_array = tf.constant([[0, 1, 2], [2, 3, 1]])
        num_tokens = 3
        expected_output_shape = (None, num_tokens)
        encoder_layer = layers.CategoryEncoding(num_tokens)
        input_data = layers.Input(shape=(3,), dtype=tf.int32)
        int_data = encoder_layer(input_data)
        self.assertAllEqual(expected_output_shape, int_data.shape)
        model = Model(inputs=input_data, outputs=int_data)
        _ = model(valid_array, training=False)
        with self.assertRaisesRegex(
            tf.errors.InvalidArgumentError,
            ".*must be in the range 0 <= values < num_tokens.*",
        ):
            _ = model(invalid_array, training=False)

    def test_dense_negative(self):
        valid_array = tf.constant([[0, 1, 2], [0, 1, 2]])
        invalid_array = tf.constant([[1, 2, 0], [2, 2, -1]])
        num_tokens = 3
        expected_output_shape = (None, num_tokens)
        encoder_layer = layers.CategoryEncoding(num_tokens)
        input_data = layers.Input(shape=(3,), dtype=tf.int32)
        int_data = encoder_layer(input_data)
        self.assertAllEqual(expected_output_shape, int_data.shape)
        model = Model(inputs=input_data, outputs=int_data)
        _ = model(valid_array, training=False)
        with self.assertRaisesRegex(
            tf.errors.InvalidArgumentError,
            ".*must be in the range 0 <= values < num_tokens.*",
        ):
            _ = model(invalid_array, training=False)

    def test_one_hot_output(self):
        input_data = np.array([[3], [2], [0], [1]])
        expected_output = [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ]
        num_tokens = 4
        expected_output_shape = (None, num_tokens)

        layer = layers.CategoryEncoding(
            num_tokens=num_tokens, output_mode="one_hot"
        )
        output_data = layer(input_data)
        self.assertAllEqual(expected_output, output_data)

        inputs = layers.Input(shape=(1,), dtype=tf.int32)
        outputs = layer(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        output_data = model(input_data)

        self.assertAllEqual(expected_output_shape, outputs.shape)
        self.assertAllEqual(expected_output, output_data)

    def test_one_hot_output_rank_one_input(self):
        input_data = np.array([3, 2, 0, 1])
        expected_output = [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ]
        num_tokens = 4
        expected_output_shape = (None, num_tokens)
        layer = layers.CategoryEncoding(
            num_tokens=num_tokens, output_mode="one_hot"
        )
        output_data = layer(input_data)
        self.assertAllEqual(expected_output, output_data)

        inputs = layers.Input(shape=(1,), dtype=tf.int32)
        outputs = layer(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        output_data = model(np.expand_dims(input_data, 0))
        self.assertAllEqual(expected_output_shape, outputs.shape)
        self.assertAllEqual(expected_output, output_data)

    def test_one_hot_output_rank_zero_input(self):
        input_data = np.array(3)
        expected_output = [[0, 0, 0, 1]]
        num_tokens = 4
        expected_output_shape = (None, num_tokens)

        layer = layers.CategoryEncoding(
            num_tokens=num_tokens, output_mode="one_hot"
        )
        output_data = layer(input_data)
        self.assertAllEqual(expected_output, output_data)

        inputs = layers.Input(shape=(), dtype=tf.int32)
        outputs = layer(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        output_data = model(np.expand_dims(input_data, 0))
        self.assertAllEqual(expected_output_shape, outputs.shape)
        self.assertAllEqual(expected_output, output_data)

    def test_one_hot_rank_3_output_fails(self):
        layer = layers.CategoryEncoding(num_tokens=4, output_mode="one_hot")
        with self.assertRaisesRegex(
            ValueError, "Maximum supported input rank*"
        ):
            _ = layer(np.array([[[3, 2, 0, 1], [3, 2, 0, 1]]]))

    def test_multi_hot_output(self):
        input_data = np.array([[1, 2, 3, 1], [0, 3, 1, 0]])
        expected_output = [
            [0, 1, 1, 1, 0, 0],
            [1, 1, 0, 1, 0, 0],
        ]
        num_tokens = 6
        expected_output_shape = (None, num_tokens)

        layer = layers.CategoryEncoding(
            num_tokens=num_tokens, output_mode="multi_hot"
        )
        output_data = layer(input_data)
        self.assertAllEqual(expected_output, output_data)

        inputs = layers.Input(shape=(4,), dtype=tf.int32)
        outputs = layer(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        output_data = model(input_data)
        self.assertAllEqual(expected_output_shape, outputs.shape)
        self.assertAllEqual(expected_output, output_data)

    def test_multi_hot_output_rank_one_input(self):
        input_data = np.array([3, 2, 0, 1])
        expected_output = [1, 1, 1, 1, 0, 0]
        num_tokens = 6
        expected_output_shape = (None, num_tokens)
        layer = layers.CategoryEncoding(
            num_tokens=num_tokens, output_mode="multi_hot"
        )
        output_data = layer(input_data)
        self.assertAllEqual(expected_output, output_data)

        # Test call on model.
        inputs = layers.Input(shape=(4,), dtype=tf.int32)
        outputs = layer(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        output_data = model(np.expand_dims(input_data, 0))[0]
        self.assertAllEqual(expected_output_shape, outputs.shape)
        self.assertAllEqual(expected_output, output_data)

    def test_multi_hot_output_rank_zero_input(self):
        input_data = np.array(3)
        expected_output = [0, 0, 0, 1, 0, 0]
        num_tokens = 6
        expected_output_shape = (num_tokens,)

        # Test call on layer directly.
        layer = layers.CategoryEncoding(
            num_tokens=num_tokens, output_mode="multi_hot"
        )
        output_data = layer(input_data)
        self.assertAllEqual(expected_output, output_data)

        # Test call on model.
        inputs = layers.Input(shape=(), dtype=tf.int32)
        outputs = layer(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        output_data = model(np.expand_dims(input_data, 0))
        self.assertAllEqual(expected_output_shape, outputs.shape)
        self.assertAllEqual(expected_output, output_data)

    def test_multi_hot_rank_3_output_fails(self):
        layer = layers.CategoryEncoding(num_tokens=4, output_mode="multi_hot")
        with self.assertRaisesRegex(
            ValueError, "Maximum supported input rank*"
        ):
            _ = layer(np.array([[[3, 2, 0, 1], [3, 2, 0, 1]]]))

    def test_count_output(self):
        input_data = np.array([[1, 2, 3, 1], [0, 3, 1, 0]])

        # pyformat: disable
        expected_output = [[0, 2, 1, 1, 0, 0], [2, 1, 0, 1, 0, 0]]
        # pyformat: enable
        num_tokens = 6
        expected_output_shape = (None, num_tokens)
        layer = layers.CategoryEncoding(num_tokens=6, output_mode="count")
        output_data = layer(input_data)
        self.assertAllEqual(expected_output, output_data)

        inputs = layers.Input(shape=(4,), dtype=tf.int32)
        outputs = layer(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        output_data = model(input_data)
        self.assertAllEqual(expected_output_shape, outputs.shape)
        self.assertAllEqual(expected_output, output_data)
