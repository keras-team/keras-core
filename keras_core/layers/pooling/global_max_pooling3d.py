from keras_core import operations as ops
from keras_core.layers.pooling.base_global_pooling import BaseGlobalPooling


class GlobalMaxPooling3D(BaseGlobalPooling):
    def __init__(self, data_format=None, keepdims=False, **kwargs):
        super().__init__(
            pool_dimensions=3,
            data_format=data_format,
            keepdims=keepdims,
            **kwargs,
        )

    def call(self, inputs):
        if self.data_format == "channels_last":
            return ops.max(inputs, axis=[1, 2, 3], keepdims=self.keepdims)
        return ops.max(inputs, axis=[2, 3, 4], keepdims=self.keepdims)
