from keras_core.optimizers.base_optimizer import BaseOptimizer


class TorchOptimizer(BaseOptimizer):
    def __new__(cls, *args, **kwargs):
        # Import locally to avoid circular imports.
        from keras_core import optimizers
        from keras_core.backend.torch import optimizers as torch_optimizers

        OPTIMIZERS = {optimizers.SGD: torch_optimizers.SGD}
        if cls in OPTIMIZERS:
            return OPTIMIZERS[cls](*args, **kwargs)
        return super().__new__(cls, *args, **kwargs)
