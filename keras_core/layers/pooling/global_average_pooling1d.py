from keras_core import operations as ops
from keras_core.layers.pooling.base_global_pooling import BaseGlobalPooling


class GlobalAveragePooling1D(BaseGlobalPooling):
    def __init__(self, data_format=None, keepdims=False, **kwargs):
        super().__init__(
            pool_dimensions=1,
            data_format=data_format,
            keepdims=keepdims,
            **kwargs,
        )

    def call(self, inputs):
        steps_axis = 1 if self.data_format == "channels_last" else 2
        return ops.mean(inputs, axis=steps_axis, keepdims=self.keepdims)
