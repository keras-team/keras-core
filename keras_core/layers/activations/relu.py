from keras_core import activations
from keras_core.api_export import keras_core_export
from keras_core.layers.layer import Layer
from keras_core import operations as ops
from keras_core import backend


@keras_core_export("keras_core.layers.ReLU")
class ReLU(Layer):
    """Applies an Exponential Linear Unit function to an output.
    It follows:
    ```
        f(x) = (exp(x) - 1.) for x < 0
        f(x) = x for x >= 0
    ```
    Args:
        **kwargs: Base layer keyword arguments, such as
            `name` and `dtype`.
    """

    def __init__(
        self, max_value=None, negative_slope=0.0, threshold=0.0, **kwargs
    ):
        super().__init__(**kwargs)
        if max_value is not None and max_value < 0.0:
            raise ValueError(
                "max_value of a ReLU layer cannot be a negative "
                f"value. Received: {max_value}"
            )
        if negative_slope is None or negative_slope < 0.0:
            raise ValueError(
                "negative_slope of a ReLU layer cannot be a negative "
                f"value. Received: {negative_slope}"
            )
        if threshold is None or threshold < 0.0:
            raise ValueError(
                "threshold of a ReLU layer cannot be a negative "
                f"value. Received: {threshold}"
            )

        self.supports_masking = True
        if max_value is not None:
            max_value = ops.cast(max_value, dtype=backend.floatx())
        self.max_value = max_value
        self.negative_slope = ops.cast(negative_slope, dtype=backend.floatx())
        self.threshold = ops.cast(threshold, dtype=backend.floatx())

    def call(self, inputs):
        return activations.relu(inputs, negative_slope=self.negative_slope, max_value=self.max_value, threshold=self.threshold)
    
    def get_config(self):
        config = super().get_config()
        config.update({"max_value": self.max_value,
                  "negative_slope": self.negative_slope,
                  "threshold": self.threshold})
        
        return config

    def compute_output_shape(self, input_shape):
        return input_shape