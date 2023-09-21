import numpy as np

from keras_core.backend.numpy import core as numpy_core
from keras_core.testing import test_case


class ConvertToTensorTest(test_case.TestCase):
    def test_convert_numpy_to_tensor(self):
        """Convert a numpy array to tensor."""
        array = np.array([1, 2, 3])
        tensor = numpy_core.convert_to_tensor(array)
        self.assertTrue(numpy_core.is_tensor(tensor))

    def test_convert_variable_to_tensor(self):
        """Convert a Variable instance to tensor."""
        value = [1, 2, 3]
        var = numpy_core.Variable(value)
        tensor = numpy_core.convert_to_tensor(var)
        self.assertTrue(numpy_core.is_tensor(tensor))
        self.assertTrue(
            np.array_equal(tensor, var.value)
        )  # Corrected this line

    def test_convert_with_dtype_float64(self):
        """Convert with specified dtype."""
        array = np.array([1, 2, 3])
        tensor = numpy_core.convert_to_tensor(
            array, dtype=np.float64
        )
        self.assertTrue(numpy_core.is_tensor(tensor))
        self.assertEqual(tensor.dtype, np.float64)

    def test_convert_with_dtype_int32(self):
        """Convert with specified dtype."""
        array = np.array([1, 2, 3])
        tensor = numpy_core.convert_to_tensor(array, dtype=np.int32)
        self.assertEqual(tensor.dtype, np.int32)

    def test_convert_sparse_error(self):
        """Test error when sparse is True."""
        with self.assertRaisesRegex(
            ValueError, "`sparse=True` is not supported with numpy backend"
        ):
            numpy_core.convert_to_tensor([1, 2, 3], sparse=True)

    def test_standardize_dtype_with_known_python_types():
        assert standardize_dtype(int) == "int64"  # or "int32" depending on the backend
        assert standardize_dtype(float) == "float32"
        assert standardize_dtype(bool) == "bool"
        assert standardize_dtype(str) == "string"

    def test_standardize_dtype_with_numpy_types():
        assert standardize_dtype(np.float32) == "float32"
        assert standardize_dtype(np.float64) == "float64"
        assert standardize_dtype(np.int32) == "int32"
        assert standardize_dtype(np.int64) == "int64"
        assert standardize_dtype(np.uint32) == "uint32"
        assert standardize_dtype(np.bool_) == "bool"

    def test_standardize_dtype_with_string_values():
        assert standardize_dtype("float32") == "float32"
        assert standardize_dtype("int64") == "int64"
        assert standardize_dtype("bool") == "bool"
        assert standardize_dtype("string") == "string"

    def test_standardize_dtype_with_unknown_types():
        with pytest.raises(ValueError, match="Invalid dtype"):
            standardize_dtype("unknown_dtype")

    def test_standardize_dtype_with_torch_types(mocker):
        # Mocking a torch dtype object for this test
        class MockTorchDtype:
            def __str__(self):
                return "torch.float32"
        mocker.patch("keras_core.backend.config.backend", return_value="torch")
        assert standardize_dtype(MockTorchDtype()) == "float32"

