import numpy as np
import torch

from keras_core import backend
from keras_core import initializers
from keras_core.backend import config
from keras_core.backend.common.variables import ALLOWED_DTYPES
from keras_core.backend.common.variables import AutocastScope
from keras_core.backend.common.variables import KerasVariable
from keras_core.backend.common.variables import standardize_dtype
from keras_core.backend.common.variables import standardize_shape
from keras_core.testing import test_case


class VariablesTest(test_case.TestCase):
    def test_deferred_initialization(self):
        """Test that variables are not initialized until they are used."""
        with backend.StatelessScope():
            v = backend.Variable(
                initializer=initializers.RandomNormal(), shape=(2, 2)
            )
            self.assertEqual(v._value, None)
            # Variables can nevertheless be accessed
            _ = v + 1
        self.assertEqual(v._value.shape, (2, 2))

        with self.assertRaisesRegex(ValueError, "while in a stateless scope"):
            with backend.StatelessScope():
                v = backend.Variable(initializer=0)

    def test_deferred_assignment(self):
        """Test that variables are not assigned until they are used."""
        with backend.StatelessScope() as scope:
            v = backend.Variable(
                initializer=initializers.RandomNormal(), shape=(2, 2)
            )
            self.assertEqual(v._value, None)
            v.assign(np.zeros((2, 2)))
            v.assign_add(2 * np.ones((2, 2)))
            v.assign_sub(np.ones((2, 2)))
        out = scope.get_current_value(v)
        self.assertAllClose(out, np.ones((2, 2)))

    def test_autocasting(self):
        """Test that variables are autocasted when used in an autocast scope"""
        v = backend.Variable(
            initializer=initializers.RandomNormal(),
            shape=(2, 2),
            dtype="float32",
        )
        self.assertEqual(v.dtype, "float32")
        self.assertEqual(backend.standardize_dtype(v.value.dtype), "float32")

        print("open scope")
        with AutocastScope("float16"):
            self.assertEqual(
                backend.standardize_dtype(v.value.dtype), "float16"
            )
        self.assertEqual(backend.standardize_dtype(v.value.dtype), "float32")

        # Test non-float variables are not affected
        v = backend.Variable(
            initializer=initializers.Ones(),
            shape=(2, 2),
            dtype="int32",
            trainable=False,
        )
        self.assertEqual(v.dtype, "int32")
        self.assertEqual(backend.standardize_dtype(v.value.dtype), "int32")

        with AutocastScope("float16"):
            self.assertEqual(backend.standardize_dtype(v.value.dtype), "int32")

    def test_standardize_dtype_with_torch_dtype(self):
        """Test that torch dtypes are converted to strings."""
        import torch

        x = torch.randn(4, 4)
        backend.standardize_dtype(x.dtype)

    def test_name_validation(self):
        """Test that variable names are validated."""
        with self.assertRaisesRegex(
            ValueError, "Argument `name` must be a string"
        ):
            KerasVariable(initializer=initializers.RandomNormal(), name=12345)

        # Test when name contains a '/'
        with self.assertRaisesRegex(ValueError, "cannot contain character `/`"):
            KerasVariable(
                initializer=initializers.RandomNormal(), name="invalid/name"
            )

    def test_standardize_shape_with_none(self):
        """Raises a ValueError when shape is None."""
        with self.assertRaisesRegex(
            ValueError, "Undefined shapes are not supported."
        ):
            standardize_shape(None)

    def test_standardize_shape_with_non_iterable(self):
        """Raises a ValueError when shape is not iterable."""
        with self.assertRaisesRegex(
            ValueError, "Cannot convert '42' to a shape."
        ):
            standardize_shape(42)

    def test_standardize_shape_with_valid_input(self):
        shape = [3, 4, 5]
        standardized_shape = standardize_shape(shape)
        self.assertEqual(standardized_shape, (3, 4, 5))


    def test_standardize_shape_with_non_integer_entry(self):
        """Raises a ValueError when shape contains non-integer entries."""
        with self.assertRaisesRegex(
            ValueError,
            "Cannot convert '\\(3, 4, 'a'\\)' to a shape. Found invalid",
        ):
            standardize_shape([3, 4, "a"])

    def test_standardize_shape_with_negative_entry(self):
        """Raises a ValueError when shape contains negative entries."""
        with self.assertRaisesRegex(
            ValueError,
            "Cannot convert '\\(3, 4, -5\\)' to a shape. Negative dimensions",
        ):
            standardize_shape([3, 4, -5])

    def test_autocast_scope_with_non_float_dtype(self):
        """Raises a ValueError when dtype is not a floating-point dtype."""
        with self.assertRaisesRegex(
            ValueError,
            "`AutocastScope` can only be used with a floating-point",
        ):
            _ = AutocastScope("int32")


class StandardizeDtypeTest(test_case.TestCase):
    def test_standardize_dtype_with_none(self):
        """Returns config.floatx() when dtype is None."""
        self.assertEqual(standardize_dtype(None), config.floatx())

    def test_standardize_dtype_with_float(self):
        """Returns "float32" when dtype is float."""
        self.assertEqual(standardize_dtype(float), "float32")

    def test_standardize_dtype_with_int(self):
        """Returns "int64" on TensorFlow, and "int32" on other backends."""
        expected = "int64" if config.backend() == "tensorflow" else "int32"
        self.assertEqual(standardize_dtype(int), expected)

    def test_standardize_dtype_with_str(self):
        """Returns "string" when dtype is str."""
        self.assertEqual(standardize_dtype(str), "string")

    def test_standardize_dtype_with_torch_dtype(self):
        """Returns "float32" when dtype is a torch.float32."""
        torch_dtype = "<class 'torch.float32'>"
        self.assertEqual(standardize_dtype(torch_dtype), "float32")

    def test_standardize_dtype_with_numpy_dtype(self):
        """Returns "float64" when dtype is a np.float64."""
        numpy_dtype = "<class 'numpy.float64'>"
        self.assertEqual(standardize_dtype(numpy_dtype), "float64")

    def test_standardize_dtype_with_invalid_dtype(self):
        """Raises a ValueError when dtype is invalid."""
        invalid_dtype = "invalid_dtype"
        with self.assertRaisesRegex(
            ValueError, f"Invalid dtype: {invalid_dtype}"
        ):
            standardize_dtype(invalid_dtype)

    def test_standardize_dtype_with_allowed_dtypes(self):
        """Returns the dtype when it is one of the allowed dtypes."""
        for dtype in ALLOWED_DTYPES:
            self.assertEqual(standardize_dtype(dtype), dtype)

    def test_standardize_dtype_with_torch_dtypes(self):
        """Returns the dtype when it is a torch dtype."""
        self.assertEqual(standardize_dtype(torch.float32), "float32")
        self.assertEqual(standardize_dtype(torch.float64), "float64")
        self.assertEqual(standardize_dtype(torch.int32), "int32")
        self.assertEqual(standardize_dtype(torch.int64), "int64")
        self.assertEqual(standardize_dtype(torch.bool), "bool")

    def test_standardize_dtype_with_numpy_dtypes(self):
        """Returns the dtype when it is a numpy dtype."""
        self.assertEqual(standardize_dtype(np.float32), "float32")
        self.assertEqual(standardize_dtype(np.float64), "float64")
        self.assertEqual(standardize_dtype(np.int32), "int32")
        self.assertEqual(standardize_dtype(np.int64), "int64")
        self.assertEqual(standardize_dtype(np.bool_), "bool_")

    def test_standardize_dtype_with_builtin_types(self):
        """Returns the dtype when it is a builtin type, such as int or float"""
        self.assertEqual(standardize_dtype(int), "int64")
        self.assertEqual(standardize_dtype(float), "float32")

    def test_standardize_dtype_with_invalid_types(self):
        """Raises a ValueError when dtype is invalid."""
        with self.assertRaises(ValueError):
            standardize_dtype("invalid_type")

    def test_standardize_dtype_with_none_input(self):
        """Returns "float32" when dtype is None."""
        self.assertEqual(standardize_dtype(None), "float32")

    def test_standardize_dtype_with_custom_types(self):
        """Raises a ValueError when dtype is a custom type."""
        class CustomType:
            pass

        with self.assertRaises(ValueError):
            standardize_dtype(CustomType)
