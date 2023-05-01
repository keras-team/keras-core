from keras_core.layers.pooling.base_global_pooling import BaseGlobalPooling

from keras_core import operations as ops

class GlobalMaxPooling1D(BaseGlobalPooling):
    def call(self, inputs):
        steps_axis = 1 if self.data_format == "channels_last" else 2
        return ops.max(inputs, axis=steps_axis, keepdims=self.keepdims)