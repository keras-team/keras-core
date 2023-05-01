from keras_core.layers.pooling.base_global_pooling import BaseGlobalPooling

from keras_core import operations as ops

class GlobalMaxPooling2D(BaseGlobalPooling):
    def call(self, inputs):
        if self.data_format == "channels_last":
            return ops.max(inputs, axis=[1, 2], keepdims=self.keepdims)
        return ops.max(inputs, axis=[2, 3], keepdims=self.keepdims)