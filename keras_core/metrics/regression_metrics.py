from keras_core import operations as ops
from keras_core.losses import losses
from keras_core.metrics import reduction_metrics
from keras_core.metrics.metric import Metric


class MeanSquaredError(reduction_metrics.MeanMetricWrapper):
    def __init__(self, name="mean_squared_error", dtype=None):
        super().__init__(fn=losses.mean_squared_error, name=name, dtype=dtype)

    def get_config(self):
        # TODO: placeholder, this should be refactored into base class
        return {"name": self.name, "dtype": self.dtype}
