class JaxLayer:
    def _post_build(self):
        """Can be overriden to perform post-build actions."""
        pass

    def _get_save_spec(self):
        if self._save_spec is None:
            return None

        from keras_core.utils.module_utils import tensorflow as tf

        def _jax_save_spec_to_tf(save_spec):
            spec = tf.TensorSpec(**save_spec)
            shape = spec.shape
            if shape.rank is None or shape.rank == 0:
                return spec

            shape_list = shape.as_list()
            shape_list[0] = None
            shape = tf.TensorShape(shape_list)
            spec._shape = shape
            return spec

        return _jax_save_spec_to_tf(self._save_spec["inputs"])
