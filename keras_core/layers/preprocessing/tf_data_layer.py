import tree

import keras_core.backend
from keras_core.layers.layer import Layer
from keras_core.random.seed_generator import SeedGenerator
from keras_core.utils import backend_utils
from keras_core.utils import tracking


class TFDataLayer(Layer):
    """Layer that can safely used in a tf.data pipeline.

    The `call()` method must solely rely on `self.backend` ops.

    Only supports a single input tensor argument.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backend = backend_utils.DynamicBackend()
        self._allow_non_tensor_positional_args = True

    def __call__(self, inputs, **kwargs):
        if backend_utils.in_tf_graph() and not isinstance(
            inputs, keras_core.KerasTensor
        ):
            # We're in a TF graph, e.g. a tf.data pipeline.
            self.backend.set_backend("tensorflow")
            inputs = tree.map_structure(
                lambda x: self.backend.convert_to_tensor(
                    x, dtype=self.compute_dtype
                ),
                inputs,
            )
            switch_convert_input_args = False
            if self._convert_input_args:
                self._convert_input_args = False
                switch_convert_input_args = True
            try:
                outputs = super().__call__(inputs, **kwargs)
            finally:
                self.backend.reset()
                if switch_convert_input_args:
                    self._convert_input_args = True
            return outputs
        return super().__call__(inputs, **kwargs)

    @tracking.no_automatic_dependency_tracking
    def _get_seed_generator(self, backend=None):
        if backend is None or backend == keras_core.backend.backend():
            return self.generator
        if not hasattr(self, "_backend_generators"):
            self._backend_generators = {}
        if backend in self._backend_generators:
            return self._backend_generators[backend]
        seed_generator = SeedGenerator(self.seed, backend=self.backend)
        self._backend_generators[backend] = seed_generator
        return seed_generator
