from keras_core import operations as ops
from keras_core.api_export import keras_core_export
from keras_core.losses.loss import squeeze_to_same_rank
from keras_core.losses.losses import log_cosh
from keras_core.losses.losses import mean_absolute_error
from keras_core.losses.losses import mean_absolute_percentage_error
from keras_core.losses.losses import mean_squared_error
from keras_core.losses.losses import mean_squared_logarithmic_error
from keras_core.metrics import reduction_metrics
from keras_core.utils.numerical_utils import normalize


@keras_core_export("keras_core.metrics.MeanSquaredError")
class MeanSquaredError(reduction_metrics.MeanMetricWrapper):
    """Computes the mean squared error between `y_true` and `y_pred`.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Example:

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


@keras_core_export("keras_core.metrics.MeanAbsoluteError")
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

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}


@keras_core_export("keras_core.metrics.MeanAbsolutePercentageError")
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

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}


@keras_core_export("keras_core.metrics.MeanSquaredLogarithmicError")
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

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}


@keras_core_export("keras_core.metrics.CosineSimilarity")
class CosineSimilarity(reduction_metrics.MeanMetricWrapper):
    """Computes the cosine similarity between the labels and predictions.

    Formula:
    `cosine similarity = (a . b) / ||a|| ||b||`
    See: [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity).
    This metric keeps the average cosine similarity between `predictions` and
    `labels` over a stream of data.

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        axis: (Optional) Defaults to -1. The dimension along which the cosine
            similarity is computed.

    Examples:

    Standalone usage:

    >>> # l2_norm(y_true) = [[0., 1.], [1./1.414, 1./1.414]]
    >>> # l2_norm(y_pred) = [[1., 0.], [1./1.414, 1./1.414]]
    >>> # l2_norm(y_true) . l2_norm(y_pred) = [[0., 0.], [0.5, 0.5]]
    >>> # result = mean(sum(l2_norm(y_true) . l2_norm(y_pred), axis=1))
    >>> #        = ((0. + 0.) +  (0.5 + 0.5)) / 2
    >>> m = tf.keras.metrics.CosineSimilarity(axis=1)
    >>> m.update_state([[0., 1.], [1., 1.]], [[1., 0.], [1., 1.]])
    >>> m.result().numpy()
    0.49999997
    >>> m.reset_state()
    >>> m.update_state([[0., 1.], [1., 1.]], [[1., 0.], [1., 1.]],
    ...                sample_weight=[0.3, 0.7])
    >>> m.result().numpy()
    0.6999999

    Usage with `compile()` API:

    ```python
    model.compile(
        optimizer='sgd',
        loss='mse',
        metrics=[tf.keras.metrics.CosineSimilarity(axis=1)])
    ```
    """

    def __init__(self, name="cosine_similarity", dtype=None, axis=-1):
        super().__init__(cosine_similarity, name, dtype=dtype, axis=axis)

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}


@keras_core_export("keras_core.metrics.LogCoshError")
class LogCoshError(reduction_metrics.MeanMetricWrapper):
    """Computes the logarithm of the hyperbolic cosine of the prediction error.

    Formula:
    `logcosh = log((exp(x) + exp(-x))/2)`, where x is the error (y_pred -
    y_true)

    Args:
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.

    Examples:

    Standalone usage:

    >>> m = tf.keras.metrics.LogCoshError()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]])
    >>> m.result().numpy()
    0.10844523
    >>> m.reset_state()
    >>> m.update_state([[0, 1], [0, 0]], [[1, 1], [0, 0]],
    ...                sample_weight=[1, 0])
    >>> m.result().numpy()
    0.21689045

    Usage with `compile()` API:

    ```python
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[tf.keras.metrics.LogCoshError()])
    ```
    """

    def __init__(self, name="logcosh", dtype=None):
        super().__init__(log_cosh, name, dtype=dtype)

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}


def cosine_similarity(y_true, y_pred, axis=-1):
    """Computes the cosine similarity between labels and predictions.

    Formula:
    ```python
    loss = sum(l2_norm(y_true) * l2_norm(y_pred))
    ```

    Args:
        y_true: Tensor of true targets.
        y_pred: Tensor of predicted targets.
        axis: Axis along which to determine similarity. Defaults to -1.

    Returns:
        Cosine similarity tensor.

    Example:

    >>> y_true = [[0., 1.], [1., 1.], [1., 1.]]
    >>> y_pred = [[1., 0.], [1., 1.], [-1., -1.]]
    >>> loss = keras_core.losses.cosine_similarity(y_true, y_pred, axis=-1)
    [0., 0.99999994, -0.99999994]
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    y_true, y_pred = squeeze_to_same_rank(y_true, y_pred)
    y_pred = normalize(y_pred, axis=axis)
    y_true = normalize(y_true, axis=axis)
    return ops.sum(y_true * y_pred, axis=axis)
