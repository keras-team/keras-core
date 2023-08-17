"""Test module for distribute_scope.py"""
import pytest

from keras_core import backend
from keras_core.backend.common import distribute_scope
from keras_core.backend.jax import distribute
from keras_core.testing import test_case


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="Only JAX backend support distribution API for now.",
)
class DistributeScopeTest(test_case.TestCase):
    def setUp(self):
        self._distribution = distribute.DataParallelDistribute()

    def test_get_distribute_scope(self):
        scope = distribute_scope.DistributeScope(self._distribution)
        scope_2 = distribute_scope.DistributeScope(self._distribution)

        self.assertIsNone(distribute_scope.get_distribute_scope())
        with scope:
            self.assertEqual(distribute_scope.get_distribute_scope(), scope)
            with scope_2:
                self.assertEqual(
                    distribute_scope.get_distribute_scope(), scope_2
                )

            self.assertEqual(distribute_scope.get_distribute_scope(), scope)

        self.assertIsNone(distribute_scope.get_distribute_scope())

    def test_in_distribute_scope(self):
        self.assertFalse(distribute_scope.in_distribute_scope())

        with distribute_scope.DistributeScope(self._distribution):
            self.assertTrue(distribute_scope.in_distribute_scope())

        self.assertFalse(distribute_scope.in_distribute_scope())

    def test_distribute_scope(self):
        scope = distribute_scope.DistributeScope(self._distribution)
        self.assertIs(scope.distribute, self._distribution)
