from keras_core import backend
from keras_core import initializers
from keras_core import operations as ops
from keras_core.api_export import keras_core_export
from keras_core.metrics.metric import Metric
from keras_core.utils import metrics_utils


class _ConfusionMatrixConditionCount(Metric):
    """Calculates the number of the given confusion matrix condition.

    Args:
      confusion_matrix_cond: One of `metrics_utils.ConfusionMatrix` conditions.
      thresholds: (Optional) Defaults to 0.5. A float value or a python
        list/tuple of float threshold values in [0, 1]. A threshold is compared
        with prediction values to determine the truth value of predictions
        (i.e., above the threshold is `true`, below is `false`). One metric
        value is generated for each threshold value.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """

    def __init__(
        self, confusion_matrix_cond, thresholds=None, name=None, dtype=None
    ):
        super().__init__(name=name, dtype=dtype)
        self._confusion_matrix_cond = confusion_matrix_cond
        self.init_thresholds = thresholds
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=0.5
        )
        self._thresholds_distributed_evenly = (
            metrics_utils.is_evenly_distributed_thresholds(self.thresholds)
        )
        self.accumulator = self.add_variable(
            shape=(len(self.thresholds),),
            initializer=initializers.Zeros(),
            name="accumulator",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the metric statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can
            be a tensor whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`.

        Returns:
          Update op.
        """
        return metrics_utils.update_confusion_matrix_variables(
            {self._confusion_matrix_cond: self.accumulator},
            y_true,
            y_pred,
            thresholds=self.thresholds,
            thresholds_distributed_evenly=self._thresholds_distributed_evenly,
            sample_weight=sample_weight,
        )

    def result(self):
        if len(self.thresholds) == 1:
            result = self.accumulator[0]
        else:
            result = self.accumulator
        return backend.convert_to_tensor(result)

    def get_config(self):
        config = {"thresholds": self.init_thresholds}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras_core_export("keras_core.metrics.FalsePositives")
class FalsePositives(_ConfusionMatrixConditionCount):
    """Calculates the number of false positives.

    If `sample_weight` is given, calculates the sum of the weights of
    false positives. This metric creates one local variable, `accumulator`
    that is used to keep track of the number of false positives.

    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.

    Args:
      thresholds: (Optional) Defaults to 0.5. A float value, or a Python
        list/tuple of float threshold values in [0, 1]. A threshold is compared
        with prediction values to determine the truth value of predictions
        (i.e., above the threshold is `true`, below is `false`). If used with a
        loss function that sets `from_logits=True` (i.e. no sigmoid applied to
        predictions), `thresholds` should be set to 0. One metric value is
        generated for each threshold value.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.

    Standalone usage:

    >>> m = keras_core.metrics.FalsePositives()
    >>> m.update_state([0, 1, 0, 0], [0, 0, 1, 1])
    >>> m.result()
    2.0

    >>> m.reset_state()
    >>> m.update_state([0, 1, 0, 0], [0, 0, 1, 1], sample_weight=[0, 0, 1, 0])
    >>> m.result()
    1.0
    """

    def __init__(self, thresholds=None, name=None, dtype=None):
        super().__init__(
            confusion_matrix_cond=metrics_utils.ConfusionMatrix.FALSE_POSITIVES,
            thresholds=thresholds,
            name=name,
            dtype=dtype,
        )
