"""Tests for JAX based distribution."""
import os

import numpy as np
import pytest

from keras_core import backend
from keras_core import layers
from keras_core import models
from keras_core import testing
from keras_core.backend.common import distribute_scope
from keras_core.backend.jax import distribute

import jax

prev_xla_flags = None


def setUpModule():
    global prev_xla_flags
    prev_xla_flags = os.getenv("XLA_FLAGS")
    flags_str = prev_xla_flags or ""
    # Don't override user-specified device count, or other XLA flags.
    if "xla_force_host_platform_device_count" not in flags_str:
        os.environ["XLA_FLAGS"] = (
            flags_str + " --xla_force_host_platform_device_count=8"
        )


def tearDownModule():
    if prev_xla_flags is None:
        del os.environ["XLA_FLAGS"]
    else:
        os.environ["XLA_FLAGS"] = prev_xla_flags


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="Only JAX backend support distribution API for now.",
)
class DataParallelDistributeTest(testing.TestCase):
    def test_create_with_devices(self):
        devices = jax.devices()
        self.assertEqual(len(devices), 8)
        distribution = distribute.DataParallelDistribute(devices=devices)

        mesh = distribution.mesh
        self.assertEqual(len(mesh.devices), 8)
        self.assertEqual(mesh.axis_names, ("batch",))
        self.assertEqual(
            distribution._data_sharding,
            jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec("batch")
            ),
        )
        self.assertEqual(
            distribution._weight_sharding,
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None)),
        )

    def test_create_with_mesh(self):
        mesh = jax.sharding.Mesh(jax.devices(), "data")
        distribution = distribute.DataParallelDistribute(mesh=mesh)
        self.assertEqual(distribution.mesh, mesh)

        self.assertEqual(
            distribution._data_sharding,
            jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec("data")
            ),
        )
        self.assertEqual(
            distribution._weight_sharding,
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None)),
        )

    def test_create_with_available_devices(self):
        distribution = distribute.DataParallelDistribute()

        mesh = distribution.mesh
        self.assertEqual(len(mesh.devices), 8)
        self.assertEqual(mesh.axis_names, ("batch",))

        self.assertEqual(
            distribution._data_sharding,
            jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec("batch")
            ),
        )
        self.assertEqual(
            distribution._weight_sharding,
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None)),
        )

    def test_mesh_with_rank_2(self):
        mesh = jax.sharding.Mesh(
            np.array(jax.devices()).reshape(4, 2), ("data", "model")
        )
        distribution = distribute.DataParallelDistribute(mesh=mesh)
        self.assertEqual(distribution.mesh, mesh)

        self.assertEqual(
            distribution._data_sharding,
            jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec("data", None)
            ),
        )
        self.assertEqual(
            distribution._weight_sharding,
            jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(None, None)
            ),
        )

    def test_distribute_data(self):
        distribution = distribute.DataParallelDistribute()

        data = np.arange(16).reshape((8, 2))
        distributed_data = distribution.distribute_data(data)
        self.assertEqual(distributed_data.sharding, distribution._data_sharding)

    def test_distribute_weight(self):
        distribution = distribute.DataParallelDistribute()

        weights = np.arange(16).reshape((8, 2))
        distributed_weights = distribution.distribute_weight(weights)
        self.assertEqual(
            distributed_weights.sharding, distribution._weight_sharding
        )

    def test_e2e_model(self):
        data_distribution = distribute.DataParallelDistribute()
        with distribute_scope.DistributeScope(data_distribution):
            inputs = layers.Input(shape=[28, 28, 1])
            y = layers.Flatten()(inputs)
            y = layers.Dense(units=200, use_bias=False, activation="relu")(y)
            y = layers.Dropout(0.4)(y)
            y = layers.Dense(units=10, activation="softmax")(y)
            model = models.Model(inputs=inputs, outputs=y)

        # Make sure all the weights are properly sharded.
        for weight in model.weights:
            self.assertEquals(
                weight._value.sharding, data_distribution._weight_sharding
            )

        # TODO(qlzh727): Need to validate the data sharding
