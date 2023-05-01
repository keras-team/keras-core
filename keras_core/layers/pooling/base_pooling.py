from keras_core import operations as ops
from keras_core.backend import KerasTensor
from keras_core.backend import image_data_format
from keras_core.layers.input_spec import InputSpec
from keras_core.layers.layer import Layer


class BasePooling(Layer):
    """Base pooling layer."""

    def __init__(
        self,
        pool_size,
        strides,
        pool_dimensions,
        pool_mode="max",
        padding="valid",
        data_format=None,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.pool_size = pool_size
        self.strides = pool_size if strides is None else strides
        self.pool_mode = pool_mode
        self.padding = padding
        self.data_format = (
            image_data_format() if data_format is None else data_format
        )

        self.input_spec = InputSpec(ndim=pool_dimensions + 2)

    def call(self, inputs):
        if self.pool_mode == "max":
            return ops.max_pool(
                inputs,
                pool_size=self.pool_size,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
            )
        elif self.pool_mode == "average":
            return ops.average_pool(
                inputs,
                pool_size=self.pool_size,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
            )
        else:
            raise ValueError(
                "`pool_mode` must be either `max` or `average`, but received "
                f"{self.pool_mode}."
            )

    def compute_output_shape(self, input_shape):
        inputs = KerasTensor(input_shape)
        # `max_pool` and `average_pool` return the same shape, so we just use
        # `ops.max_pool` for computation.
        outputs = ops.max_pool(
            inputs,
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
        )
        return outputs.shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pool_size": self.pool_size,
                "padding": self.padding,
                "strides": self.strides,
                "data_format": self.data_format,
            }
        )
        return config
