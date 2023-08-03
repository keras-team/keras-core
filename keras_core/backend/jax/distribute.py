"""!!!DO NOT USE!!!

Distribution related class for JAX backend.

This is just a prototype and we might want to unify it in future for other
backends.
"""

from absl import logging
import jax
import numpy as np

_DEFAULT_BATCH_DIM_NAME = "batch"


class DataParallelDistribute:
    def __init__(self, mesh=None, devices=None):
        """Create the data parallel distribution.

        User can choose to create this instance by either `Mesh` or `devices`
        parameters (but not both).

        The mesh is expected to be a `jax.sharding.Mesh` instance, and is expected
        to be 1D only. In case that the mesh has multiple axises, then the first
        axis will be treated as data parallel dimension (and a warning will be
        raised).

        When a list of `devices` are provided, they will be used to construct a 1D
        mesh.

        When both `mesh` and `devices` are absent, then we will rely on
        `jax.devices` to detect any available devices, and create mesh from them.
        """
        super().__init__()
        if mesh:
            self._init_with_mesh(mesh)
        elif devices:
            self._init_mesh_from_devices(devices)
        else:
            self._init_mesh_from_jax_devices()

        self._config_sharding_spec()
        self._batch_dim_name = self.mesh.axis_names[0]

    def distribute_data(self, data):
        return jax.device_put(data, self._data_sharding)

    def distribute_weight(self, weight):
        return jax.device_put(weight, self._weight_sharding)

    def _init_with_mesh(self, mesh):
        if not isinstance(mesh, jax.sharding.Mesh):
            raise ValueError(
                "Expect the mesh to be type of jax.sharding.Mesh, "
                f"Received {type(mesh)}"
            )
        self._user_provide_devices = None
        self.mesh = mesh
        if self.mesh.devices.ndim != 1:
            logging.warning(
                "Expect the input mesh to be 1D, but received %dD. "
                "The first axis will be used for data parallel sharding",
                self.mesh.devices.ndim,
            )

    def _init_mesh_from_devices(self, devices):
        self._user_provide_devices = devices
        self.mesh = jax.sharding.Mesh(
            np.array(devices), _DEFAULT_BATCH_DIM_NAME
        )

    def _init_mesh_from_jax_devices(self):
        self._user_provide_devices = None
        self.mesh = jax.sharding.Mesh(
            np.array(jax.devices()), _DEFAULT_BATCH_DIM_NAME
        )

    def _config_sharding_spec(self):
        weight_shard_spec = [None] * self.mesh.devices.ndim  # Fully replicated
        data_shard_spec = weight_shard_spec.copy()
        data_shard_spec[0] = self.mesh.axis_names[0]  # Shard on the first dim

        self._data_sharding = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec(*data_shard_spec)
        )
        self._weight_sharding = jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec(*weight_shard_spec)
        )
