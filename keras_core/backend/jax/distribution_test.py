"""Tests for JAX based distribution."""
import os

import jax
from jax._src import xla_bridge as xb
import numpy as np
import pytest

from keras_core import backend
from keras_core import layers
from keras_core import models
from keras_core import testing
from keras_core.backend.common import global_state
from keras_core.backend.jax import distribution

prev_xla_flags = None

prev_xla_flags = os.getenv("XLA_FLAGS")
flags_str = prev_xla_flags or ""
# Don't override user-specified device count, or other XLA flags.
if "xla_force_host_platform_device_count" not in flags_str:
    os.environ["XLA_FLAGS"] = (
        flags_str + " --xla_force_host_platform_device_count=8"
    )
# Clear any cached backends so new CPU backend will pick up the env var.
xb.get_backend.cache_clear()

devices = jax.devices()
assert len(devices) == 8, f"{devices} should have 8 devices."


# def setUpModule():
    # global prev_xla_flags



def tearDownModule():
    if prev_xla_flags is None:
        del os.environ["XLA_FLAGS"]
    else:
        os.environ["XLA_FLAGS"] = prev_xla_flags
    # Clear any cached backends so new CPU backend will pick up the env var.
    xb.get_backend.cache_clear()

    devices = jax.devices()
    assert len(devices) == 1, f"{devices} should only have 1 item after cleanup"


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="Only JAX backend support distribution API for now.",
)
class DataParallelDistributionTest(testing.TestCase):
    def test_create_with_devices(self):
        devices = jax.devices()
        self.assertEqual(len(devices), 8)
        ds = distribution.DataParallelDistribution(devices=devices)

        mesh = ds.mesh
        self.assertEqual(len(mesh.devices), 8)
        self.assertEqual(mesh.axis_names, ("batch",))
        self.assertEqual(
            ds._data_sharding,
            jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec("batch")
            ),
        )
        self.assertEqual(
            ds._variable_sharding,
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None)),
        )

    def test_create_with_mesh(self):
        mesh = jax.sharding.Mesh(jax.devices(), "data")
        ds = distribution.DataParallelDistribution(mesh=mesh)
        self.assertEqual(ds.mesh, mesh)

        self.assertEqual(
            ds._data_sharding,
            jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec("data")
            ),
        )
        self.assertEqual(
            ds._variable_sharding,
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None)),
        )

    def test_create_with_available_devices(self):
        ds = distribution.DataParallelDistribution()

        mesh = ds.mesh
        self.assertEqual(len(mesh.devices), 8)
        self.assertEqual(mesh.axis_names, ("batch",))

        self.assertEqual(
            ds._data_sharding,
            jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec("batch")
            ),
        )
        self.assertEqual(
            ds._variable_sharding,
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None)),
        )

    def test_mesh_with_rank_2(self):
        mesh = jax.sharding.Mesh(
            np.array(jax.devices()).reshape(4, 2), ("data", "model")
        )
        ds = distribution.DataParallelDistribution(mesh=mesh)
        self.assertEqual(ds.mesh, mesh)

        self.assertEqual(
            ds._data_sharding,
            jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec("data", None)
            ),
        )
        self.assertEqual(
            ds._variable_sharding,
            jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(None, None)
            ),
        )

    def test_distribute_data(self):
        ds = distribution.DataParallelDistribution()

        data = np.arange(16).reshape((8, 2))
        distributed_data = ds.distribute_data(data)
        self.assertEqual(distributed_data.sharding, ds._data_sharding)

    def test_distribute_variable(self):
        ds = distribution.DataParallelDistribution()

        weights = np.arange(16).reshape((8, 2))
        distributed_weights = ds.distribute_variable(weights)
        self.assertEqual(distributed_weights.sharding, ds._variable_sharding)

    def test_scope(self):
        self.assertIsNone(distribution.get_distribution())
        data_distribution = distribution.DataParallelDistribution()
        with data_distribution.scope():
            self.assertIs(distribution.get_distribution(), data_distribution)
            data_distribution_2 = distribution.DataParallelDistribution()
            with data_distribution_2.scope():
                self.assertIs(
                    distribution.get_distribution(), data_distribution_2
                )

            self.assertIs(distribution.get_distribution(), data_distribution)

        self.assertIsNone(distribution.get_distribution())

    def test_as_global_distribution(self):
        try:
            self.assertIsNone(distribution.get_distribution())

            data_distribution = distribution.DataParallelDistribution()
            data_distribution.as_global_distribution()
            self.assertIs(distribution.get_distribution(), data_distribution)
        finally:
            # Cleanup the global state
            global_state.set_global_attribute(
                distribution._GLOBAL_ATTRIBUTE_NAME, None
            )

    def test_e2e_model(self):
        data_distribution = distribution.DataParallelDistribution()
        with data_distribution.scope():
            inputs = layers.Input(shape=[28, 28, 1])
            y = layers.Flatten()(inputs)
            y = layers.Dense(units=200, use_bias=False, activation="relu")(y)
            y = layers.Dropout(0.4)(y)
            y = layers.Dense(units=10, activation="softmax")(y)
            model = models.Model(inputs=inputs, outputs=y)

        # Make sure all the weights are properly sharded.
        for weight in model.weights:
            self.assertEqual(
                weight._value.sharding, data_distribution._variable_sharding
            )

        # TODO(qlzh727): Need to validate the data sharding
