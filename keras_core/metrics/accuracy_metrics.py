from keras_core import operations as ops
from keras_core.metrics import reduction_metrics
from keras_core.api_export import keras_core_export
from keras_core.backend import floatx


def accuracy(y_true, y_pred):
    y_true = ops.convert_to_tensor(y_true, dtype=floatx())
    y_pred = ops.convert_to_tensor(y_pred, dtype=floatx())
    return ops.cast(ops.equal(y_true, y_pred), dtype=floatx())


@keras_core_export("keras_core.metrics.Accuracy")
class Accuracy(reduction_metrics.MeanMetricWrapper):
    """Calculates how often predictions equal labels.

    This metric creates two local variables, `total` and `count` that are used
    to compute the frequency with which `y_pred` matches `y_true`. This
    frequency is ultimately returned as `binary accuracy`: an idempotent
    operation that simply divides `total` by `count`.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = keras_core.metrics.Accuracy()
    >>> m.update_state([[1], [2], [3], [4]], [[0], [2], [3], [4]])
    >>> m.result()
    0.75

    >>> m.reset_state()
    >>> m.update_state([[1], [2], [3], [4]], [[0], [2], [3], [4]],
    ...                sample_weight=[1, 1, 0, 0])
    >>> m.result()
    0.5

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[keras_core.metrics.Accuracy()])
    ```
    """

    def __init__(self, name="accuracy", dtype=None):
        super().__init__(fn=accuracy, name=name, dtype=dtype)

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}