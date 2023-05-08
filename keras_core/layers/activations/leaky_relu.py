from keras_core import activations
from keras_core.api_export import keras_core_export
from keras_core.layers.layer import Layer


@keras_core_export("keras_core.layers.LeakyReLU")
class LeakyReLU(Layer):
    """Leaky version of a Rectified Linear Unit activation layer.

    This layer allows a small gradient when the unit is not active.

    Formula:

    ``` python
    f(x) = alpha * x if x < 0
    f(x) = x if x >= 0
    ```

    Example:

    ``` python
    leaky_relu_layer = LeakyReLU(negative_slope=0.5)
    input = np.array([-10, -5, 0.0, 5, 10])
    result = leaky_relu_layer(input)
    # result = [-5. , -2.5,  0. ,  5. , 10.]
    ```

    Args:
        negative_slope: Float >= 0.0. Negative slope coefficient.
          Defaults to 0.3.
        **kwargs: Base layer keyword arguments, such as
            `name` and `dtype`.

    """

    def __init__(self, negative_slope=0.3, **kwargs):
        super().__init__(**kwargs)
        if negative_slope is None:
            raise ValueError(
                "The negative_slope value of a Leaky ReLU layer "
                "cannot be None, Expecting a float. Received: "
                f"negative_slope={negative_slope}"
            )
        self.supports_masking = True
        self.negative_slope = negative_slope

    def call(self, inputs):
        return activations.leaky_relu(
            inputs, negative_slope=self.negative_slope
        )

    def get_config(self):
        config = super().get_config()
        config.update({"negative_slope": self.negative_slope})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
