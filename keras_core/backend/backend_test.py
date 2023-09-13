import importlib
from unittest.mock import patch

from keras_core import backend as backend_module
from keras_core.testing import test_case


# Define a test case class for testing the backend module
class BackendTestCase(test_case.TestCase):
    # Helper function to reload the backend module with a mocked backend function
    def reload_backend_module_with_mocked_backend(self, mocked_backend_value):
        """
        Helper function to reload the backend module with a mocked backend function.
        """
        with patch(
            "keras_core.backend.config.backend",
            return_value=mocked_backend_value,
        ):
            importlib.reload(backend_module)

    # Test for JAX backend function import
    def test_jax_backend_function_import(self):
        # Reload the backend module with a mocked JAX backend
        self.reload_backend_module_with_mocked_backend("jax")

        # Check for the presence of convert_to_tensor in the imported backend module
        self.assertTrue(hasattr(backend_module.core, "convert_to_tensor"))
        self.assertTrue(
            hasattr(backend_module.backend.jax.core, "convert_to_tensor")
        )

    # Test for TensorFlow backend function import
    def test_tensorflow_backend_function_import(self):
        # Reload the backend module with a mocked TensorFlow backend
        self.reload_backend_module_with_mocked_backend("tensorflow")

        # Check for the presence of convert_to_tensor in the imported backend module
        self.assertTrue(hasattr(backend_module.core, "convert_to_tensor"))
        self.assertTrue(
            hasattr(backend_module.backend.tensorflow.core, "convert_to_tensor")
        )

    # Test for NumPy backend function import
    def test_numpy_backend_function_import(self):
        # Reload the backend module with a mocked NumPy backend
        self.reload_backend_module_with_mocked_backend("numpy")

        # Check for the presence of convert_to_tensor in the imported backend module
        self.assertTrue(hasattr(backend_module.core, "convert_to_tensor"))
        self.assertTrue(
            hasattr(backend_module.backend.numpy.core, "convert_to_tensor")
        )

    # Test for PyTorch backend function import
    def test_torch_backend_function_import(self):
        # Reload the backend module with a mocked PyTorch backend
        self.reload_backend_module_with_mocked_backend("torch")

        # Check for the presence of convert_to_tensor in the imported backend module
        self.assertTrue(hasattr(backend_module.core, "convert_to_tensor"))
        self.assertTrue(
            hasattr(backend_module.backend.torch.core, "convert_to_tensor")
        )

    # Test for an invalid backend, expecting a specific ValueError message
    def test_invalid_backend(self):
        with self.assertRaisesRegex(
            ValueError, "Unable to import backend : invalid_backend"
        ):
            # Attempt to reload the backend module with an invalid backend name
            self.reload_backend_module_with_mocked_backend("invalid_backend")
