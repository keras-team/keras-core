"""Unified high level distribution APIs across backends.

!!!DO NOT USE!!! Currently under development and APIs are not final.

Currently only the JAX backend has been implemented, and the Tensorflow backend
will be implemented in future (via tf.dtensor API).
"""

import contextlib
from typing import Any

from keras_core import KerasVariable
from keras_core.backend.common import global_state

GLOBAL_ATTRIBUTE_NAME = "distribution"


def list_devices(device_type: str = None) -> list[str]:
    """Return all the available devices based on the device type.

    Note that this should return the global devices in a distributed setting.

    Args:
        device_type: CPU, GPU or TPU. Default to GPU or TPU if available when
            device_type is not provided. Otherwise will return the CPU devices.

    Return:
        List of devices that are available for distribute computation.
    """
    pass


class DeviceMesh:
    """The cluster of computation devices for distributed computation.

    This is aligned with `jax.sharding.Mesh` and `tf.dtensor.Mesh`, which
    represents the computation devices in the global context.

    See more details in
    https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.Mesh
    and https://www.tensorflow.org/api_docs/python/tf/experimental/dtensor/Mesh.
    """

    def __init__(
        self,
        shape: list | tuple,
        axis_names: list[str],
        devices: list[str] = None,
    ):
        """Initialize the DeviceMesh for the given topology.

        Args:
            shape: the shape of the overall DeviceMesh, e.g. (8,) for a data
                parallel only distribution, or (4, 2) for a model+data parallel
                distribution.
            axis_names: The logical name of the each axis for the DeviceMesh.
                The length of the `axis_names` should match to the rank of the
                `shape`. The `axis_names` will be used to match/create the
                `TensorLayout` when distribute the data and weights.
            devices: Optional list of devices. Default to all the available
                devices locally from `list_devices()`.
        """
        pass


class TensorLayout:
    """The layout of a Tensor.

    This is aligned with `jax.sharding.NamedSharding` and `tf.dtensor.Layout`,
    which allocate the tensor to its logic axis based on the `DeviceMesh`. With
    `DeviceMesh` and `TensorLayout`, the actual mapping between a Tensor to the
    physical devices can be determined.

    See more details in
    https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.NamedSharding
    and https://www.tensorflow.org/api_docs/python/tf/experimental/dtensor/Layout.  # noqa: E501
    """

    def __init__(self, layout_spec: list | tuple):
        """Initialize the TensorLayout with layout_spec.

        Args:
            layout_spec: list of strings that should map to the `axis_names` in
                `DeviceMesh`. For any dimentions that doesn't need any sharding,
                A `None` can be used a placeholder.
        """
        pass


class Distribution:
    """Base class for the distribution.

    The `Distribution` has following key functionalities.

    1. Distribute the model variables to the `DeviceMesh`.
    2. Distribute the input data to the `DeviceMesh`.

    It can create a context scope so that the framework to properly detect the
    `Distribution` and distribute the variable/data accordingly.
    """

    def __init__(self, device_mesh: DeviceMesh):
        pass

    def distribute_data(self, data) -> Any:
        """Shard the input data based on the Distribution setting.

        Args:
            data: input data, can be Tensor and np.Array.

        Returns:
            distributed data tensor.
        """
        pass

    def distribute_variable(self, variable: KerasVariable) -> KerasVariable:
        """Distribute the variable based on the Distribution setting.

        Args:
            variable: a keras variable for distribution.

        return:
            distributed keras variable.
        """
        pass

    def as_global_distribution(self):
        """Set the current `Distribution` as the global distribution setting."""
        pass

    @contextlib.contextmanager
    def scope(self):
        """Context manager to make the `Distribution` current."""
        pass


def get_global_distribution():
    """Retrieve the current distribution from global context."""
    return global_state.get_global_attribute(GLOBAL_ATTRIBUTE_NAME)


def set_global_distribution(distribution: Distribution):
    """Set the distribution as the global distribution setting."""
    distribution.as_global_distribution()
