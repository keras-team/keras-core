from keras_core.backend.jax import distribution


class JaxLayer:
    def _post_build(self):
        """Can be overriden to perform post-build actions."""
        if distribution.get_distribution() is None:
            return

        distribute = distribution.get_distribution()
        # Swap both trainable/non-trainable weights,
        for v in self._trainable_variables:
            v.assign(distribute.distribute_variable(v))
        for v in self._non_trainable_variables:
            v.assign(distribute.distribute_variable(v))

        # We don't care about the sub layers, which will be handled within the
        # _post_build() by itself.
