from keras_core.backend.common.backend_utils import (
    _convert_conv_tranpose_padding_args_from_keras_to_jax,
)
from keras_core.backend.common.backend_utils import (
    _convert_conv_tranpose_padding_args_from_keras_to_torch,
)
from keras_core.backend.common.backend_utils import (
    _get_output_shape_given_tf_padding,
)
from keras_core.backend.common.backend_utils import (
    compute_conv_transpose_padding_args_for_jax,
)
from keras_core.backend.common.backend_utils import (
    compute_conv_transpose_padding_args_for_torch,
)
from keras_core.testing import test_case


class ConvertConvTransposePaddingArgsJAXTest(test_case.TestCase):
    def test_valid_padding_without_output_padding(self):
        """Test conversion with 'valid' padding and no output padding."""
        (
            left_pad,
            right_pad,
        ) = _convert_conv_tranpose_padding_args_from_keras_to_jax(
            kernel_size=3,
            stride=2,
            dilation_rate=1,
            padding="valid",
            output_padding=None,
        )
        self.assertEqual(left_pad, 2)
        self.assertEqual(right_pad, 2)

    def test_same_padding_without_output_padding(self):
        """Test conversion with 'same' padding and no output padding."""
        (
            left_pad,
            right_pad,
        ) = _convert_conv_tranpose_padding_args_from_keras_to_jax(
            kernel_size=3,
            stride=2,
            dilation_rate=1,
            padding="same",
            output_padding=None,
        )
        self.assertEqual(left_pad, 2)
        self.assertEqual(right_pad, 1)

    def test_invalid_padding_type(self):
        """Test with an invalid padding type."""
        with self.assertRaises(AssertionError):
            _convert_conv_tranpose_padding_args_from_keras_to_jax(
                kernel_size=3,
                stride=2,
                dilation_rate=1,
                padding="unknown",
                output_padding=None,
            )


class ConvertConvTransposePaddingArgsTorchTest(test_case.TestCase):
    def test_valid_padding_without_output_padding(self):
        """Test conversion with 'valid' padding and no output padding"""
        (
            torch_padding,
            torch_output_padding,
        ) = _convert_conv_tranpose_padding_args_from_keras_to_torch(
            kernel_size=3,
            stride=2,
            dilation_rate=1,
            padding="valid",
            output_padding=None,
        )
        # Expected values based on the function's logic
        self.assertEqual(torch_padding, 0)
        self.assertEqual(torch_output_padding, 0)

    def test_same_padding_without_output_padding(self):
        """Test conversion with 'same' padding and no output padding"""
        (
            torch_padding,
            torch_output_padding,
        ) = _convert_conv_tranpose_padding_args_from_keras_to_torch(
            kernel_size=3,
            stride=2,
            dilation_rate=1,
            padding="same",
            output_padding=None,
        )
        # Expected values based on the function's logic
        self.assertEqual(torch_padding, 1)
        self.assertEqual(torch_output_padding, 1)

    def test_invalid_padding_type(self):
        """Test with an invalid padding type"""
        with self.assertRaises(AssertionError):
            _convert_conv_tranpose_padding_args_from_keras_to_torch(
                kernel_size=3,
                stride=2,
                dilation_rate=1,
                padding="unknown",
                output_padding=None,
            )


class ComputeConvTransposePaddingArgsForJAXTest(test_case.TestCase):
    def test_valid_padding_without_output_padding(self):
        """Test computation with 'valid' padding and no output padding"""
        jax_padding = compute_conv_transpose_padding_args_for_jax(
            input_shape=(1, 5, 5, 3),
            kernel_shape=(3, 3, 3, 3),
            strides=2,
            padding="valid",
            output_padding=None,
            dilation_rate=1,
        )
        self.assertEqual(jax_padding, [(2, 2), (2, 2)])

    def test_same_padding_without_output_padding(self):
        """Test computation with 'same' padding and no output padding"""
        jax_padding = compute_conv_transpose_padding_args_for_jax(
            input_shape=(1, 5, 5, 3),
            kernel_shape=(3, 3, 3, 3),
            strides=2,
            padding="same",
            output_padding=None,
            dilation_rate=1,
        )

        self.assertEqual(jax_padding, [(2, 1), (2, 1)])


class ComputeConvTransposePaddingArgsForTorchTest(test_case.TestCase):
    def test_valid_padding_without_output_padding(self):
        """Test computation with 'valid' padding and no output padding"""
        (
            torch_paddings,
            torch_output_paddings,
        ) = compute_conv_transpose_padding_args_for_torch(
            input_shape=(1, 5, 5, 3),
            kernel_shape=(3, 3, 3, 3),
            strides=2,
            padding="valid",
            output_padding=None,
            dilation_rate=1,
        )
        self.assertEqual(torch_paddings, [0, 0])
        self.assertEqual(torch_output_paddings, [0, 0])

    def test_same_padding_without_output_padding(self):
        """Test computation with 'same' padding and no output padding"""
        (
            torch_paddings,
            torch_output_paddings,
        ) = compute_conv_transpose_padding_args_for_torch(
            input_shape=(1, 5, 5, 3),
            kernel_shape=(3, 3, 3, 3),
            strides=2,
            padding="same",
            output_padding=None,
            dilation_rate=1,
        )
        self.assertEqual(torch_paddings, [1, 1])
        self.assertEqual(torch_output_paddings, [1, 1])


class GetOutputShapeGivenTFPaddingTest(test_case.TestCase):
    def test_valid_padding_without_output_padding(self):
        """Test computation with 'valid' padding and no output padding."""
        output_shape = _get_output_shape_given_tf_padding(
            input_size=5,
            kernel_size=3,
            strides=2,
            padding="valid",
            output_padding=None,
            dilation_rate=1,
        )
        self.assertEqual(output_shape, 11)

    def test_same_padding_without_output_padding(self):
        """Test computation with 'same' padding and no output padding."""
        output_shape = _get_output_shape_given_tf_padding(
            input_size=5,
            kernel_size=3,
            strides=2,
            padding="same",
            output_padding=None,
            dilation_rate=1,
        )
        self.assertEqual(output_shape, 10)

    def test_valid_padding_with_output_padding(self):
        """Test computation with 'valid' padding and output padding."""
        output_shape = _get_output_shape_given_tf_padding(
            input_size=5,
            kernel_size=3,
            strides=2,
            padding="valid",
            output_padding=1,
            dilation_rate=1,
        )
        self.assertEqual(output_shape, 12)
