"""!!!DO NOT USE!!!

Distribution related class for JAX backend.

This is just a prototype and we might want to unify it
with other backends in the future.
"""
import jax


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
    device_type = device_type.lower() if device_type else None
    return jax.devices(backend=device_type)


def to_jax_mesh(device_mesh):
    """Convert the DeviceMesh to JAX backend specific Mesh.

    Args:
        device_mesh: DeviceMesh instance to convert.

    Returns:
        A `jax.sharding.Mesh` instance.
    """
    return jax.sharding.Mesh(device_mesh.devices, device_mesh.axis_names)


def to_jax_layout(tensor_layout):
    """Convert the TensorLayout to JAX backend specific Sharding.

    Args:
        tensor_layout: TensorLayout instance to convert.

    Returns:
        A `jax.sharding.NamedSharding` instance.
    """
    if tensor_layout.device_mesh is None:
        raise ValueError(
            "Cannot create sharding when device mesh is not set "
            "for TensorLayout."
        )
    partition_spec = jax.sharding.PartitionSpec(*tensor_layout.axes)
    jax_mesh = to_jax_mesh(tensor_layout.device_mesh)
    return jax.sharding.NamedSharding(jax_mesh, partition_spec)


def distribute_data(data, distribution):
    """Distribute the input data based on the distribution setting.

    Args:
        data: `jax.Array` or a nested structure `jax.Array` to distribution
        distribution: `keras_core.distribution.Distribution` instance. Could be
            `None`.
    Returns:
        Distributed data. In the case that `distribution` is None, the original
        data will be returned.
    """
    if distribution is None:
        return data

    jax_sharding = jax.tree_util.tree_map(
        lambda d: to_jax_layout(distribution.get_data_layout(d.shape)), data)
    return jax.device_put(data, jax_sharding)


def distribute_variable(value, variable_path, distribution):
    """Distribute the variable init value based on the distribution setting.

    Args:
        value: `jax.Array`, the init value for distribution.
        variable_path: str, the path of the variable, which is obtained from
            name scope.
        distribution: `keras_core.distribution.Distribution` instance. Could be
            `None`.
    Returns:
        Distributed variable. In the case that `distribution` is None, 
        the original value will be returned.
    """
    if distribution is None:
        return value

    variable_sharding = to_jax_layout(
        distribution.get_variable_layout(value.shape, variable_path))
    return jax.device_put(value, variable_sharding)
