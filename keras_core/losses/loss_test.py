import numpy as np

from keras_core import operations as ops
from keras_core import testing
from keras_core.losses import MeanAbsoluteError
from keras_core.losses.loss import Loss


class ExampleLoss(Loss):
    def call(self, y_true, y_pred):
        return (y_true - y_pred) ** 2


class LossTest(testing.TestCase):
    def test_reduction(self):
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        # No reduction
        loss_fn = ExampleLoss(reduction=None)
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(loss.dtype.name, "float32")
        self.assertAllClose((y_true - y_pred) ** 2, loss)

        # sum
        loss_fn = ExampleLoss(reduction="sum")
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(loss.dtype.name, "float32")
        self.assertAllClose(np.sum((y_true - y_pred) ** 2), loss)

        # sum_over_batch_size
        loss_fn = ExampleLoss(reduction="sum_over_batch_size")
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(loss.dtype.name, "float32")
        self.assertAllClose(np.sum((y_true - y_pred) ** 2) / 4, loss)

        # bad reduction
        with self.assertRaisesRegex(ValueError, "Invalid value for argument"):
            ExampleLoss(reduction="abc")

    def test_mask(self):
        mask = np.array([True, False, True, True])
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        masked_y_true = np.array([1.0, 1.0, 0.0])
        masked_y_pred = np.array([0.1, 0.3, 0.4])

        mask = ops.convert_to_tensor(mask)
        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        y_pred._keras_mask = mask

        loss_fn = ExampleLoss()
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(loss.dtype.name, "float32")
        self.assertAllClose(
            np.sum((masked_y_true - masked_y_pred) ** 2) / 3, loss
        )

        # Test edge case where everything is masked.
        mask = np.array([False, False, False, False])
        y_pred._keras_mask = mask
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(loss.dtype.name, "float32")
        self.assertEqual(loss, 0)  # No NaN.

    def test_sample_weight(self):
        sample_weight = np.array([0.4, 0.3, 0.2, 0.1])
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        loss_fn = ExampleLoss()
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(loss.dtype.name, "float32")
        self.assertAllClose(
            np.sum(sample_weight * (y_true - y_pred) ** 2) / 4, loss
        )

        # Test edge case where every weight is 0.
        sample_weight = np.array([0.0, 0.0, 0.0, 0.0])
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(loss.dtype.name, "float32")
        self.assertEqual(loss, 0)  # No NaN.

    def test_mask_and_sample_weight(self):
        sample_weight = np.array([0.4, 0.3, 0.2, 0.1])
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])
        mask = np.array([True, False, True, True])

        masked_sample_weight = np.array([0.4, 0.2, 0.1])
        masked_y_true = np.array([1.0, 1.0, 0.0])
        masked_y_pred = np.array([0.1, 0.3, 0.4])

        mask = ops.convert_to_tensor(mask)
        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        y_pred._keras_mask = mask

        loss_fn = ExampleLoss()
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(loss.dtype.name, "float32")
        self.assertAllClose(
            np.sum(masked_sample_weight * (masked_y_true - masked_y_pred) ** 2)
            / 3,
            loss,
        )

    # @testing.parametrize("uprank", ["mask", "sample_weight", "y_true", "y_pred"])
    # TODO: use parameterization decorator
    def test_rank_adjustment(self):
        for uprank in ["mask", "sample_weight", "ys"]:
            sample_weight = np.array([0.4, 0.3, 0.2, 0.1])
            y_true = np.array([1.0, 0.0, 1.0, 0.0])
            y_pred = np.array([0.1, 0.2, 0.3, 0.4])
            mask = np.array([True, False, True, True])

            if uprank == "mask":
                mask = np.expand_dims(mask, -1)
            elif uprank == "sample_weight":
                sample_weight = np.expand_dims(sample_weight, -1)
            elif uprank == "ys":
                y_true = np.expand_dims(y_true, -1)
                y_pred = np.expand_dims(y_pred, -1)

            masked_sample_weight = np.array([0.4, 0.2, 0.1])
            masked_y_true = np.array([1.0, 1.0, 0.0])
            masked_y_pred = np.array([0.1, 0.3, 0.4])

            mask = ops.convert_to_tensor(mask)
            y_true = ops.convert_to_tensor(y_true)
            y_pred = ops.convert_to_tensor(y_pred)
            y_pred._keras_mask = mask

            loss_fn = ExampleLoss()
            loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
            self.assertEqual(loss.dtype.name, "float32")
            self.assertAllClose(
                np.sum(
                    masked_sample_weight * (masked_y_true - masked_y_pred) ** 2
                )
                / 3,
                loss,
            )

    def test_mixed_dtypes(self):
        sample_weight = np.array([0.4, 0.3, 0.2, 0.1], dtype="float64")
        y_true = np.array([1.0, 0.0, 1.0, 0.0], dtype="int32")
        y_pred = np.array([0.1, 0.2, 0.3, 0.4], dtype="float32")

        loss_fn = ExampleLoss()
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(loss.dtype.name, "float32")
        self.assertAllClose(
            np.sum(sample_weight * (y_true - y_pred) ** 2) / 4,
            loss,
        )

    def test_serialization(self):
        # TODO
        pass


class MeanAbsoluteErrorTest(testing.TestCase):
    def test_reduction(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[1.0, 1.0], [1.0, 0.0]])

        # No reduction
        loss_fn = MeanAbsoluteError(reduction=None)
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(loss.dtype.name, "float32")
        self.assertAllClose(np.mean(np.abs((y_true - y_pred)), axis=-1), loss)

        # sum
        loss_fn = MeanAbsoluteError(reduction="sum")
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(loss.dtype.name, "float32")
        self.assertAllClose(
            np.sum(np.mean(np.abs((y_true - y_pred)), axis=-1)), loss
        )

        # sum_over_batch_size
        loss_fn = MeanAbsoluteError(reduction="sum_over_batch_size")
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(loss.dtype.name, "float32")
        self.assertAllClose(
            np.sum(np.mean(np.abs((y_true - y_pred)), axis=-1)) / 2, loss
        )

        # bad reduction
        with self.assertRaisesRegex(ValueError, "Invalid value for argument"):
            MeanAbsoluteError(reduction="abc")

    def test_sample_weight(self):
        sample_weight = np.array([0.7, 0.3])
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[1.0, 1.0], [1.0, 0.0]])

        loss_fn = MeanAbsoluteError()
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(loss.dtype.name, "float32")
        self.assertAllClose(
            np.sum(sample_weight * np.mean(np.abs((y_true - y_pred)), axis=-1))
            / 2,
            loss,
        )

        # Test for sample_weight=0.0
        sample_weight = np.array([0.0, 0.0])
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(loss.dtype.name, "float32")
        self.assertEqual(loss, 0)  # No NaN.

    def test_mask(self):
        sample_mask = np.array([True, False, True])

        y_true = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]])

        masked_y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        masked_y_pred = np.array([[1.0, 1.0], [1.0, 0.0]])

        mask = ops.convert_to_tensor(sample_mask)
        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        y_pred._keras_mask = mask

        loss_fn = MeanAbsoluteError()
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(loss.dtype.name, "float32")
        self.assertAllClose(
            np.sum(np.mean(np.abs((masked_y_true - masked_y_pred)), axis=-1))
            / 2,
            loss,
        )

        # Test edge case where everything is masked.
        mask = np.array([False, False, False])
        y_pred._keras_mask = mask
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(loss.dtype.name, "float32")
        self.assertEqual(loss, 0)  # No NaN.

    def test_mask_and_sample_weight(self):
        sample_mask = np.array([True, False, True])
        sample_weight = np.array([0.7, 0.0, 0.3])

        y_true = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]])

        masked_y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        masked_y_pred = np.array([[1.0, 1.0], [1.0, 0.0]])
        masked_sample_weight = np.array([0.7, 0.3])

        mask = ops.convert_to_tensor(sample_mask)
        masked_sample_weight = ops.convert_to_tensor(masked_sample_weight)
        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        y_pred._keras_mask = mask

        loss_fn = MeanAbsoluteError()
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(loss.dtype.name, "float32")
        self.assertAllClose(
            np.sum(
                masked_sample_weight
                * np.mean(np.abs((masked_y_true - masked_y_pred)), axis=-1)
            )
            / 2,
            loss,
        )

        # Test edge case where everything is masked.
        mask = np.array([False, False, False])
        y_pred._keras_mask = mask
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(loss.dtype.name, "float32")
        self.assertEqual(loss, 0)  # No NaN.

    def test_mixed_dtypes(self):
        sample_weight = np.array([0.7, 0.3], dtype=np.float64)
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.int32)
        y_pred = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)

        loss_fn = MeanAbsoluteError()
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(loss.dtype.name, "float32")
        self.assertAllClose(
            np.sum(sample_weight * np.mean(np.abs((y_true - y_pred)), axis=-1))
            / 2,
            loss,
        )

    def test_serialization(self):
        # TODO
        pass
