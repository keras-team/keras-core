from unittest.mock import patch

from keras_core.testing import test_case
from keras_core.utils import jax_utils


class TestJaxUtils(test_case.TestCase):
    @patch("keras_core.utils.jax_utils.backend")
    def test_not_in_jax_tracing_scope_when_backend_is_not_jax(
        self, mock_backend
    ):
        mock_backend.backend.return_value = "not_jax"
        self.assertFalse(jax_utils.is_in_jax_tracing_scope())

    @patch("keras_core.utils.jax_utils.backend")
    def test_not_in_jax_tracing_and_no_tracer_detected(self, mock_backend):
        mock_backend.backend.return_value = "jax"
        mock_backend.numpy.ones.return_value = (
            object()
        )  # Some other non-tracer class
        self.assertFalse(jax_utils.is_in_jax_tracing_scope())
