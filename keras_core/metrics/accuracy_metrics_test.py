import numpy as np

from keras_core import testing
from keras_core.metrics.accuracy_metrics import Accuracy


class AccuracyTest(testing.TestCase):
    def test_accuracy(self):
        acc_obj = Accuracy(name="my_acc", dtype="float32")

        # check config
        self.assertEqual(acc_obj.name, "my_acc")
        self.assertEqual(len(acc_obj.variables), 2)
        self.assertEqual(acc_obj.dtype, "float32")

        # verify that correct value is returned
        acc_obj.update_state(
            y_true=[[1], [2], [3], [4]],
            y_pred=[[1], [2], [3], [4]],
            sample_weight=None,
        )
        result = acc_obj.result()
        self.assertEqual(result, 1)

        # check with sample_weight
        acc_obj.update_state(
            y_true=[[2], [1]], y_pred=[[2], [0]], sample_weight=[[0.5], [0.2]]
        )
        result = acc_obj.result()
        self.assertAlmostEqual(result, 0.9, 2)
