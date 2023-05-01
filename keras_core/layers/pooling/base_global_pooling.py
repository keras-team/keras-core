from keras_core import operations as ops
from keras_core.backend import KerasTensor
from keras_core.backend import image_data_format
from keras_core.layers.input_spec import InputSpec
from keras_core.layers.layer import Layer


class BaseGlobalPooling(Layer):
    """Base global pooling layer."""

    def __init__(self, pool_dimensions, data_format=None, keepdims=False, **kwargs):
        super().__init__(**kwargs)
        
        self.data_format = (
            image_data_format() if data_format is None else data_format
        )
        self.keepdims = keepdims
        self.input_spec = InputSpec(ndim=pool_dimensions + 2)

    def call(self, inputs):
        raise NotImplementedError

    def compute_output_shape(self, input_shape):
        num_spatial_dims = len(input_shape) - 2
        if self.data_format == "channels_last":
            if self.keepdims:
                return (input_shape[0],) + (1,) * num_spatial_dims + (input_shape[-1],)
            else:
                return (input_shape[0],) + (input_shape[-1],)
        else:
            if self.keepdims:
                return (input_shape[0], input_shape[1]) + (1,) * num_spatial_dims
            else:
                return (input_shape[0], input_shape[1])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "data_format": self.data_format,
                "keepdims": self.keepdims,
            }
        )
        return config
