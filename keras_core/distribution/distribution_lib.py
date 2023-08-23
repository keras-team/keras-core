"""Unified high level distribution APIs across backends.

!!!DO NOT USE!!! Currently under development and APIs are not final.

Currently only the JAX backend has been implemented, and the Tensorflow backend
will be implemented in future (via tf.dtensor API).
"""

import logging

import contextlib
import numpy as np

from keras_core.backend import distribution_lib
from keras_core.backend.common import global_state

DEFAULT_BATCH_DIM_NAME = "batch"
GLOBAL_ATTRIBUTE_NAME = "distribution"


def list_devices(device_type=None):
    """Return all the available devices based on the device type.

    Note that this should return the global devices in a distributed setting.

    Args:
        device_type: string of `"cpu"`, `"gpu"` or `"tpu"`. Default to `gpu` or
            `tpu` if available when device_type is not provided. Otherwise
            will return the `cpu` devices.

    Return:
        List of devices that are available for distribute computation.
    """
    return distribution_lib.list_devices(device_type)


class DeviceMesh:
    """The cluster of computation devices for distributed computation.

    This is aligned with `jax.sharding.Mesh` and `tf.dtensor.Mesh`, which
    represents the computation devices in the global context.

    See more details in [jax.sharding.Mesh](
        https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.Mesh)
    and [tf.dtensor.Mesh](
        https://www.tensorflow.org/api_docs/python/tf/experimental/dtensor/Mesh).
    """

    def __init__(
        self,
        shape,
        axis_names,
        devices=None,
    ):
        """Initialize the DeviceMesh for the given topology.

        Args:
            shape: tuple of list of integers. The shape of the overall
                DeviceMesh, e.g. `(8,)` for a data parallel only distribution,
                or `(4, 2)` for a model+data parallel distribution.
            axis_names: List of string. The logical name of the each axis for
                the DeviceMesh. The length of the `axis_names` should match to
                the rank of the `shape`. The `axis_names` will be used to
                match/create the `TensorLayout` when distribute the data and
                weights.
            devices: Optional list of devices. Default to all the available
                devices locally from `list_devices()`.
        """
        if not shape or not axis_names:
            raise ValueError(
                "Shape and axis_names cannot be empty. Got "
                f"shape: {shape}, axis_names: {axis_names}"
            )

        if len(shape) != len(axis_names):
            raise ValueError(
                "Shape and axis_names should have same size,"
                f"got shape: {shape} and axis_names: {axis_names}"
            )
        if devices is None:
            devices = list_devices()
        devices = np.array(devices)
        if np.prod(shape) != np.prod(devices.shape):
            raise ValueError(
                "Shape does not match the number of devices. "
                f"Got shape: {shape}, and shape of the "
                f"devices: {devices.shape}"
            )

        self._shape = shape
        self._axis_names = axis_names
        self._devices = np.reshape(devices, shape)

    @property
    def shape(self):
        return self._shape

    @property
    def axis_names(self):
        return self._axis_names

    @property
    def devices(self):
        return self._devices


class TensorLayout:
    """The layout of a Tensor.

    This is aligned with `jax.sharding.NamedSharding` and `tf.dtensor.Layout`,
    which allocate the tensor to its logic axis based on the `DeviceMesh`. With
    `DeviceMesh` and `TensorLayout`, the actual mapping between a Tensor to the
    physical devices can be determined.

    See more details in [jax.sharding.NamedSharding](
        https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.NamedSharding)
    and [tf.dtensor.Layout](
        https://www.tensorflow.org/api_docs/python/tf/experimental/dtensor/Layout).
    """

    def __init__(self, axes, device_mesh=None):
        """Initialize the TensorLayout with axis names.

        Args:
            axes: list of strings that should map to the `axis_names` in
                `DeviceMesh`. For any dimentions that doesn't need any sharding,
                A `None` can be used a placeholder.
            device_mesh: Optional `DeviceMesh` that will be used to create
                the layout. The actual mapping of tensor to physical device
                is not known until the mesh is specified.
        """
        self._axes = axes
        self._device_mesh = device_mesh
        self._validate_axes()

    @property
    def axes(self):
        return self._axes

    @property
    def device_mesh(self):
        return self._device_mesh

    @device_mesh.setter
    def device_mesh(self, device_mesh):
        if self._device_mesh is not None:
            raise ValueError(
                "Cannot override device mesh value. Existing "
                f"value is {self._device_mesh}"
            )
        self._device_mesh = device_mesh
        self._validate_axes()

    def _validate_axes(self):
        if self._device_mesh:
            valid_axis_names = set(self._device_mesh.axis_names)
            axis_names = set(self._axes) - set([None])
            if axis_names - valid_axis_names:
                raise ValueError(
                    "Invalid axis names for Layout. Valid axis "
                    f"names: {valid_axis_names}, Got {axis_names}"
                )


class Distribution:
    """Base class for the distribution.

    The `Distribution` has following key functionalities.

    1. Distribute the model variables to the `DeviceMesh`.
    2. Distribute the input data to the `DeviceMesh`.

    It can create a context scope so that the framework to properly detect the
    `Distribution` and distribute the variable/data accordingly.
    """

    def __init__(self, device_mesh):
        self._device_mesh = device_mesh

    def get_data_layout(self, data_shape):
        """Retrieve the `TensorLayout` for the input data.

        Args:
            data_shape: shape for the input data in list or tuple format.

        Returns:
            The `TensorLayout` for the data, which can be used by
            `backend.distribute_tensor()` to redistribute a input data.
        """
        raise NotImplementedError()

    def get_variable_layout(self, variable_shape, variable_path):
        """Retrieve the `TensorLayout` for the variable based on the path.

        The path of the variable is available by `variable.path`.

        Args:
            variable_shape, shape for the variable in list or tuple format.
            variable_path: string, the path for the variable to be distributed.

        return:
            The `TensorLayout` for the variable, which can be used by
            `backend.distribute_tensor()` to redistribute a variable.
        """
        raise NotImplementedError()

    @contextlib.contextmanager
    def scope(self):
        """Context manager to make the `Distribution` current."""
        original_scope = distribution()
        set_distribution(self)
        try:
            yield
        finally:
            set_distribution(original_scope)

    @property
    def device_mesh(self):
        return self._device_mesh


class DataParallel(Distribution):
    def __init__(self, device_mesh=None, devices=None):
        """Create the data parallel distribution.

        You can choose to create this instance by either specifing
        the `device_mesh` or `devices` parameters (but not both).

        The device_mesh is expected to be a `DeviceMesh` instance, and is
        expected to be 1D only. In case that the mesh has multiple axes, then
        the first axis will be treated as the data parallel dimension
        (and a warning will be raised).

        When a list of `devices` are provided, they will be used to construct a
        1D mesh.

        When both `mesh` and `devices` are absent, then `list_devices()`
        will be used to detect any available devices and create a 1D mesh from 
        them.
        """
        if device_mesh:
            self._initialize_with_device_mesh(device_mesh)
        elif devices:
            self._initialize_mesh_from_devices(devices)
        else:
            self._initialize_mesh_from_list_devices()

        self._batch_dim_name = self.device_mesh.axis_names[0]

    def _initialize_with_device_mesh(self, device_mesh):
        if not isinstance(device_mesh, DeviceMesh):
            raise ValueError(
                "Expect `mesh` to be an instance of `DeviceMesh`. "
                f"Received: mesh={device_mesh} (of type {type(device_mesh)})"
            )
        super().__init__(device_mesh)
        if self.device_mesh.devices.ndim != 1:
            logging.warning(
                "Expect the input mesh to be 1D, but received "
                "mesh.devices.ndim=%d. "
                "The first axis will be used for data-parallel sharding.",
                device_mesh.devices.ndim
            )

    def _initialize_mesh_from_devices(self, devices):
        devices = np.array(devices)
        device_mesh = DeviceMesh(
            shape=devices.shape,
            axis_names=[DEFAULT_BATCH_DIM_NAME],
            devices=devices)
        super().__init__(device_mesh)

    def _initialize_mesh_from_list_devices(self):
        devices = np.array(list_devices())
        device_mesh = DeviceMesh(
            shape=devices.shape,
            axis_names=[DEFAULT_BATCH_DIM_NAME],
            devices=devices)
        super().__init__(device_mesh)

    def get_data_layout(self, data_shape):
        data_shard_spec = [None] * len(data_shape)
        data_shard_spec[0] = self._batch_dim_name   # Shard on the first dim
        return TensorLayout(data_shard_spec, self.device_mesh)

    def get_variable_layout(self, variable_shape, variable_path):
        del variable_path
        variable_shard_spec =  [None] * len(variable_shape)
        return TensorLayout(variable_shard_spec, self.device_mesh)

def distribution():
    """Retrieve the current distribution from global context."""
    return global_state.get_global_attribute(GLOBAL_ATTRIBUTE_NAME)


def set_distribution(value):
    """Set the distribution as the global distribution setting.

    Args:
        value: a `Distribution` instance.
    """
    global_state.set_global_attribute(GLOBAL_ATTRIBUTE_NAME, value)
