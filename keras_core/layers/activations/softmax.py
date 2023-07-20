from keras_core import activations
from keras_core import backend
from keras_core.api_export import keras_core_export
from keras_core.layers.layer import Layer


def _large_negative_number(dtype):
    """Return a Large negative number based on dtype."""
    if backend.standardize_dtype(dtype) == "float16":
        return -3e4
    return -1e9


@keras_core_export("keras_core.layers.Softmax")
class Softmax(Layer):
    """Softmax activation layer.

    Formula:
    ``` python
    exp_x = exp(x - max(x))
    f(x) = exp_x / sum(exp_x)
    ```

    Example:
    >>>softmax_layer = keras_core.layers.activations.Softmax()
    >>>input = np.array([1.0, 2.0, 1.0])
    >>>result = softmax_layer(input)
    [0.21194157, 0.5761169, 0.21194157]


    Args:
        axis: Integer, or list of Integers, axis along which the softmax
            normalization is applied.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.

    Call arguments:
        inputs: The inputs (logits) to the softmax layer.
        mask: A boolean mask of the same shape as `inputs`. The mask
            specifies 1 to keep and 0 to mask. Defaults to `None`.

    Returns:
        Softmaxed output with the same shape as `inputs`.
    """

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.built = True

    def call(self, inputs, mask=None):
        if mask is not None:
            adder = (
                1.0 - backend.cast(mask, inputs.dtype)
            ) * _large_negative_number(inputs.dtype)
            inputs += adder
        if isinstance(self.axis, (tuple, list)):
            if len(self.axis) > 1:
                return backend.numpy.exp(
                    inputs
                    - backend.math.logsumexp(
                        inputs, axis=self.axis, keepdims=True
                    )
                )
            else:
                return activations.softmax(inputs, axis=self.axis[0])
        return activations.softmax(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
