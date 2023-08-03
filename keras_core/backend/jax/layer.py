from keras_core.backend.common import distribute_scope


class JaxLayer:
    def _post_build(self):
        """Can be overriden to perform post-build actions."""
        if not distribute_scope.in_distribute_scope():
            return

        distribution = distribute_scope.get_distribute_scope().distribute
        # Swap both trainable/non-trainable weights,
        for v in self._trainable_variables:
            v.assign(distribution.distribute_weight(v))
        for v in self._non_trainable_variables:
            v.assign(distribution.distribute_weight(v))

        # We don't care about the sub layers, which will be handled within the
        # _post_build() by itself.
