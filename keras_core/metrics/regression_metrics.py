from keras_core.api_export import keras_core_export
from keras_core.losses.losses import mean_squared_error
from keras_core.losses.losses import mean_absolute_error
from keras_core.losses.losses import mean_absolute_percentage_error
from keras_core.losses.losses import mean_squared_logarithmic_error
from keras_core.metrics import reduction_metrics


@keras_core_export("keras_core.metrics.MeanSquaredError")
class MeanSquaredError(reduction_metrics.MeanMetricWrapper):
    """Computes the mean squared error between `y_true` and `y_pred`.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = keras_core.metrics.MeanSquaredError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result()
    0.25
    """

    def __init__(self, name="mean_squared_error", dtype=None):
        super().__init__(fn=mean_squared_error, name=name, dtype=dtype)

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}


@keras_core_export("keras.metrics.MeanAbsoluteError")
class MeanAbsoluteError(reduction_metrics.MeanMetricWrapper):
    """Computes the mean absolute error between the labels and predictions.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:

    Standalone usage:

    >>> m = tf.keras.metrics.MeanAbsoluteError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result().numpy()
    0.25
    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    0.5

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.MeanAbsoluteError()])
    ```
    """

    def __init__(self, name="mean_absolute_error", dtype=None):
        super().__init__(mean_absolute_error, name, dtype=dtype)

@keras_core_export("keras.metrics.MeanAbsolutePercentageError")
class MeanAbsolutePercentageError(reduction_metrics.MeanMetricWrapper):
    """Computes the mean absolute percentage error between `y_true` and
    `y_pred`.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:

    Standalone usage:

    >>> m = tf.keras.metrics.MeanAbsolutePercentageError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result().numpy()
    250000000.0
    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    500000000.0

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])
    ```
    """

    def __init__(self, name="mean_absolute_percentage_error", dtype=None):
        super().__init__(mean_absolute_percentage_error, name, dtype=dtype)

@keras_core_export("keras.metrics.MeanSquaredLogarithmicError")
class MeanSquaredLogarithmicError(reduction_metrics.MeanMetricWrapper):
    """Computes the mean squared logarithmic error between `y_true` and
    `y_pred`.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:

    Standalone usage:

    >>> m = tf.keras.metrics.MeanSquaredLogarithmicError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result().numpy()
    0.12011322
    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    0.24022643

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.MeanSquaredLogarithmicError()])
    ```
    """

    def __init__(self, name="mean_squared_logarithmic_error", dtype=None):
        super().__init__(mean_squared_logarithmic_error, name, dtype=dtype)