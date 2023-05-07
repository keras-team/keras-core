"""Tests for tf.distribute related functionality under tf implementation."""

import numpy as np
import tensorflow as tf

from keras_core import backend
from keras_core import layers
from keras_core import testing


class DistributeTest(testing.TestCase):

    def setUp(self):
        super().setUp()
        if backend.backend() != 'tensorflow':
            raise ValueError('The distribute test can only run with TF backend, '
                             f'current backend is {backend.backend()}')
        # Need at least 2 devices for distribution related tests.
        cpus = tf.config.list_physical_devices("CPU")
        tf.config.set_logical_device_configuration(
            cpus[0],
            [
                tf.config.LogicalDeviceConfiguration(),
                tf.config.LogicalDeviceConfiguration(),
            ],
        )

    def test_variable_creation(self):
        pass