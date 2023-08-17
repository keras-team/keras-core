"""!!!DO NOT USE!!! Under construction and experiment."""


from keras_core.backend.common import global_state


class DistributeScope:
    def __init__(self, distribute):
        """Create a DistributeScope with a distribute mechanism.

        Args:
            distribute: A keras_core.backend.jax.distribute.Distribute
        """

        self._distribute = distribute
        self._original_scope = None

    @property
    def distribute(self):
        return self._distribute

    def __enter__(self):
        self._original_scope = get_distribute_scope()
        global_state.set_global_attribute("distribute_scope", self)
        return self

    def __exit__(self, *args, **kwargs):
        global_state.set_global_attribute(
            "distribute_scope", self._original_scope
        )


def in_distribute_scope():
    return global_state.get_global_attribute("distribute_scope") is not None


def get_distribute_scope():
    return global_state.get_global_attribute("distribute_scope")
