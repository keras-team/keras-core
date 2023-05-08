"""Tests for tf.distribute related functionality under tf implementation."""

import tensorflow as tf

from keras_core import backend
from keras_core import layers
from keras_core import models
from keras_core import testing


class DistributeTest(testing.TestCase):

    def setUp(self):
        super().setUp()
        if backend.backend() != 'tensorflow':
            raise ValueError(
                'The distribute test can only run with TF backend, '
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
        strategy = tf.distribute.MirroredStrategy(['CPU:0', 'CPU:1'])
        with strategy.scope():
            dense = layers.Dense(2)
            dense.build([4, 2])

        self.assertIsInstance(dense.kernel, backend.KerasVariable)
        self.assertIsInstance(dense.kernel.value, 
                              tf.distribute.DistributedValues)
        self.assertIn('MirroredVariable', dense.kernel.value.__class__.__name__)

        self.assertIsInstance(dense.kernel, backend.KerasVariable)
        self.assertIsInstance(dense.bias.value, tf.distribute.DistributedValues)
        self.assertIn('MirroredVariable', dense.bias.value.__class__.__name__)

    def test_strategy_run(self):
        strategy = tf.distribute.MirroredStrategy(['CPU:0', 'CPU:1'])

        with strategy.scope():
            inputs = layers.Input(shape=[4])
            dense = layers.Dense(2)
            output = dense(inputs)
            model = models.Functional(inputs, output)

        self.assertIsInstance(dense.kernel, backend.KerasVariable)
        self.assertIsInstance(dense.kernel.value, 
                              tf.distribute.DistributedValues)

        def input_fn(ctx):
            if ctx.replica_id_in_sync_group == 1:
                return tf.ones([8, 4])
            else:
                return tf.zeros([8, 4])

        distributed_inputs = strategy.experimental_distribute_values_from_function(
            input_fn)

        @tf.function
        def run_fn(data):
            return model(data)

        result = strategy.run(run_fn, args=(distributed_inputs,))

        self.assertIsInstance(result, 
                              tf.types.experimental.distributed.PerReplica)
        self.assertLen(result.values, 2)
        self.assertEqual(result.values[0].shape, [8, 2])
        self.assertEqual(result.values[1].shape, [8, 2])
        self.assertNotAllClose(result.values[0], result.values[1])
        self.assertAllClose(result.values[0], tf.zeros([8, 2]))
