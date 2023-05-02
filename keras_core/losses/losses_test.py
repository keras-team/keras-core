import numpy as np

from keras_core import testing
from keras_core.losses import losses


class MeanSquaredErrorTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(losses.MeanSquaredError(name="mymse"))

    def test_all_correct_unweighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = np.array([[4, 8, 12], [8, 1, 3]])
        loss = mse_obj(y_true, y_true)
        self.assertAlmostEqual(loss, 0.0)

    def test_unweighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mse_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 49.5)

    def test_scalar_weighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mse_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 113.85)

    def test_sample_weighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        sample_weight = np.array([[1.2], [3.4]])
        loss = mse_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 767.8 / 6)

    def test_timestep_weighted(self):
        # TODO
        pass

    def test_zero_weighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mse_obj(y_true, y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0)

    def test_invalid_sample_weight(self):
        # TODO
        pass

    def test_no_reduction(self):
        mse_obj = losses.MeanSquaredError(reduction=None)
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mse_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, [84.3333, 143.3666])

    def test_sum_reduction(self):
        mse_obj = losses.MeanSquaredError(reduction="sum")
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mse_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 227.69998)


class MeanAbsoluteErrorTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.MeanAbsoluteError(name="mymae")
        )

    def test_all_correct_unweighted(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = np.array([[4, 8, 12], [8, 1, 3]])
        loss = mae_obj(y_true, y_true)
        self.assertAlmostEqual(loss, 0.0)

    def test_unweighted(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mae_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 5.5)

    def test_scalar_weighted(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mae_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 12.65)

    def test_sample_weighted(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        sample_weight = np.array([[1.2], [3.4]])
        loss = mae_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 81.4 / 6)

    def test_timestep_weighted(self):
        # TODO
        pass

    def test_zero_weighted(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mae_obj(y_true, y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0)

    def test_invalid_sample_weight(self):
        # TODO
        pass

    def test_no_reduction(self):
        mae_obj = losses.MeanAbsoluteError(reduction=None)
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mae_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, [10.7333, 14.5666])

    def test_sum_reduction(self):
        mae_obj = losses.MeanAbsoluteError(reduction="sum")
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mae_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 25.29999)


class MeanAbsolutePercentageErrorTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.MeanAbsolutePercentageError(name="mymape")
        )

    def test_all_correct_unweighted(self):
        mape_obj = losses.MeanAbsolutePercentageError()
        y_true = np.array([[4, 8, 12], [8, 1, 3]])
        loss = mape_obj(y_true, y_true)
        self.assertAlmostEqual(loss, 0.0)

    def test_unweighted(self):
        mape_obj = losses.MeanAbsolutePercentageError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mape_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 211.8518, 3)

    def test_scalar_weighted(self):
        mape_obj = losses.MeanAbsolutePercentageError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mape_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 487.259, 3)

    def test_sample_weighted(self):
        mape_obj = losses.MeanAbsolutePercentageError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        sample_weight = np.array([[1.2], [3.4]])
        loss = mape_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 422.8888, 3)

    def test_timestep_weighted(self):
        # TODO
        pass

    def test_zero_weighted(self):
        mape_obj = losses.MeanAbsolutePercentageError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mape_obj(y_true, y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_no_reduction(self):
        mape_obj = losses.MeanAbsolutePercentageError(reduction=None)
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mape_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, [621.8518, 352.6666])


class MeanSquaredLogarithmicErrorTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.MeanSquaredLogarithmicError(name="mysloge")
        )

    def test_unweighted(self):
        msle_obj = losses.MeanSquaredLogarithmicError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = msle_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 1.4370, 3)

    def test_scalar_weighted(self):
        msle_obj = losses.MeanSquaredLogarithmicError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = msle_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 3.3051, 3)

    def test_sample_weighted(self):
        msle_obj = losses.MeanSquaredLogarithmicError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        sample_weight = np.array([[1.2], [3.4]])
        loss = msle_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 3.7856, 3)

    def test_timestep_weighted(self):
        # TODO
        pass

    def test_zero_weighted(self):
        msle_obj = losses.MeanSquaredLogarithmicError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = msle_obj(y_true, y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0, 3)


class HingeTest(testing.TestCase):
    def test_unweighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])

        # Reduction = "sum_over_batch_size"
        hinge_obj = losses.Hinge(reduction="sum_over_batch_size")
        loss = hinge_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 1.3, 3)

        # Reduction = "sum"
        hinge_obj = losses.Hinge(reduction="sum")
        loss = hinge_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 2.6, 3)

        # Reduction = None
        hinge_obj = losses.Hinge(reduction=None)
        loss = hinge_obj(y_true, y_pred)
        self.assertAllClose(loss, [1.1, 1.5])

        # Bad reduction
        with self.assertRaisesRegex(ValueError, "Invalid value for argument"):
            losses.Hinge(reduction="abc")

    def test_weighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        sample_weight = [1, 0]

        # Reduction = "sum_over_batch_size"
        hinge_obj = losses.Hinge(reduction="sum_over_batch_size")
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.55, 3)

        # Reduction = "sum"
        hinge_obj = losses.Hinge(reduction="sum")
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 1.1, 3)

        # Reduction = None
        hinge_obj = losses.Hinge(reduction=None)
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, [1.1, 0.0])

    def test_zero_weighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        sample_weight = 0.0

        hinge_obj = losses.Hinge()
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(loss, 0.0)


class SquaredHingeTest(testing.TestCase):
    def test_unweighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])

        # Reduction = "sum_over_batch_size"
        hinge_obj = losses.SquaredHinge(reduction="sum_over_batch_size")
        loss = hinge_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 1.86, 3)

        # Reduction = "sum"
        hinge_obj = losses.SquaredHinge(reduction="sum")
        loss = hinge_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 3.72, 3)

        # Reduction = None
        hinge_obj = losses.SquaredHinge(reduction=None)
        loss = hinge_obj(y_true, y_pred)
        self.assertAllClose(loss, [1.46, 2.26])

        # Bad reduction
        with self.assertRaisesRegex(ValueError, "Invalid value for argument"):
            losses.SquaredHinge(reduction="abc")

    def test_weighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        sample_weight = [1, 0]

        # Reduction = "sum_over_batch_size"
        hinge_obj = losses.SquaredHinge(reduction="sum_over_batch_size")
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.73, 3)

        # Reduction = "sum"
        hinge_obj = losses.SquaredHinge(reduction="sum")
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 1.46, 3)

        # Reduction = None
        hinge_obj = losses.SquaredHinge(reduction=None)
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, [1.46, 0.0])

    def test_zero_weighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        sample_weight = 0.0

        hinge_obj = losses.SquaredHinge()
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(loss, 0.0)


class CategoricalHingeTest(testing.TestCase):
    def test_unweighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])

        # Reduction = "sum_over_batch_size"
        hinge_obj = losses.CategoricalHinge(reduction="sum_over_batch_size")
        loss = hinge_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 1.4, 3)

        # Reduction = "sum"
        hinge_obj = losses.CategoricalHinge(reduction="sum")
        loss = hinge_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 2.8, 3)

        # Reduction = None
        hinge_obj = losses.CategoricalHinge(reduction=None)
        loss = hinge_obj(y_true, y_pred)
        self.assertAllClose(loss, [1.2, 1.6])

        # Bad reduction
        with self.assertRaisesRegex(ValueError, "Invalid value for argument"):
            losses.CategoricalHinge(reduction="abc")

    def test_weighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        sample_weight = [1, 0]

        # Reduction = "sum_over_batch_size"
        hinge_obj = losses.CategoricalHinge(reduction="sum_over_batch_size")
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.6, 3)

        # Reduction = "sum"
        hinge_obj = losses.CategoricalHinge(reduction="sum")
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 1.2, 3)

        # Reduction = None
        hinge_obj = losses.CategoricalHinge(reduction=None)
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, [1.2, 0.0])

    def test_zero_weighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        sample_weight = 0.0

        hinge_obj = losses.CategoricalHinge()
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(loss, 0.0)


class CosineSimilarityTest(testing.TestCase):
    def l2_norm(self, x, axis):
        epsilon = 1e-12
        square_sum = np.sum(np.square(x), axis=axis, keepdims=True)
        x_inv_norm = 1 / np.sqrt(np.maximum(square_sum, epsilon))
        return np.multiply(x, x_inv_norm)

    def setup(self, axis=1):
        self.np_y_true = np.asarray([[1, 9, 2], [-5, -2, 6]], dtype=np.float32)
        self.np_y_pred = np.asarray([[4, 8, 12], [8, 1, 3]], dtype=np.float32)

        y_true = self.l2_norm(self.np_y_true, axis)
        y_pred = self.l2_norm(self.np_y_pred, axis)
        self.expected_loss = np.sum(np.multiply(y_true, y_pred), axis=(axis,))

        self.y_true = self.np_y_true
        self.y_pred = self.np_y_pred

    def test_config(self):
        cosine_obj = losses.CosineSimilarity(
            axis=2, reduction="sum", name="cosine_loss"
        )
        self.assertEqual(cosine_obj.name, "cosine_loss")
        self.assertEqual(cosine_obj.reduction, "sum")

    def test_unweighted(self):
        self.setup()
        cosine_obj = losses.CosineSimilarity()
        loss = cosine_obj(self.y_true, self.y_pred)
        expected_loss = -np.mean(self.expected_loss)
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_scalar_weighted(self):
        self.setup()
        cosine_obj = losses.CosineSimilarity()
        sample_weight = 2.3
        loss = cosine_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        expected_loss = -np.mean(self.expected_loss * sample_weight)
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_sample_weighted(self):
        self.setup()
        cosine_obj = losses.CosineSimilarity()
        sample_weight = np.asarray([1.2, 3.4])
        loss = cosine_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        expected_loss = -np.mean(self.expected_loss * sample_weight)
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_timestep_weighted(self):
        self.setup()
        cosine_obj = losses.CosineSimilarity()
        np_y_true = self.np_y_true.reshape((2, 3, 1))
        np_y_pred = self.np_y_pred.reshape((2, 3, 1))
        sample_weight = np.asarray([3, 6, 5, 0, 4, 2]).reshape((2, 3))

        y_true = self.l2_norm(np_y_true, 2)
        y_pred = self.l2_norm(np_y_pred, 2)
        expected_loss = np.sum(np.multiply(y_true, y_pred), axis=(2,))

        y_true = np_y_true
        y_pred = np_y_pred
        loss = cosine_obj(y_true, y_pred, sample_weight=sample_weight)

        expected_loss = -np.mean(expected_loss * sample_weight)
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_zero_weighted(self):
        self.setup()
        cosine_obj = losses.CosineSimilarity()
        loss = cosine_obj(self.y_true, self.y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_axis(self):
        self.setup(axis=1)
        cosine_obj = losses.CosineSimilarity(axis=1)
        loss = cosine_obj(self.y_true, self.y_pred)
        expected_loss = -np.mean(self.expected_loss)
        self.assertAlmostEqual(loss, expected_loss, 3)


class HuberLossTest(testing.TestCase):
    def huber_loss(self, y_true, y_pred, delta=1.0):
        error = y_pred - y_true
        abs_error = np.abs(error)

        quadratic = np.minimum(abs_error, delta)
        linear = np.subtract(abs_error, quadratic)
        return np.add(
            np.multiply(0.5, np.multiply(quadratic, quadratic)),
            np.multiply(delta, linear),
        )

    def setup(self, delta=1.0):
        self.np_y_pred = np.array([[0.9, 0.2, 0.2], [0.8, 0.4, 0.6]])
        self.np_y_true = np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

        self.batch_size = 6
        self.expected_losses = self.huber_loss(
            self.np_y_true, self.np_y_pred, delta
        )

        self.y_pred = self.np_y_pred
        self.y_true = self.np_y_true

    def test_config(self):
        h_obj = losses.Huber(reduction="sum", name="huber")
        self.assertEqual(h_obj.name, "huber")
        self.assertEqual(h_obj.reduction, "sum")

    def test_all_correct(self):
        self.setup()
        h_obj = losses.Huber()
        loss = h_obj(self.y_true, self.y_true)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_unweighted(self):
        self.setup()
        h_obj = losses.Huber()
        loss = h_obj(self.y_true, self.y_pred)
        actual_loss = np.sum(self.expected_losses) / self.batch_size
        self.assertAlmostEqual(loss, actual_loss, 3)

    def test_scalar_weighted(self):
        self.setup()
        h_obj = losses.Huber()
        sample_weight = 2.3
        loss = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        actual_loss = (
            sample_weight * np.sum(self.expected_losses) / self.batch_size
        )
        self.assertAlmostEqual(loss, actual_loss, 3)

        # Verify we get the same output when the same input is given
        loss_2 = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, loss_2, 3)

    def test_sample_weighted(self):
        self.setup()
        h_obj = losses.Huber()
        sample_weight = np.array([[1.2], [3.4]])

        loss = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        actual_loss = np.multiply(
            self.expected_losses,
            np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3)),
        )
        actual_loss = np.sum(actual_loss) / self.batch_size
        self.assertAlmostEqual(loss, actual_loss, 3)

    def test_timestep_weighted(self):
        self.setup()
        h_obj = losses.Huber()
        y_pred = self.np_y_pred.reshape((2, 3, 1))
        y_true = self.np_y_true.reshape((2, 3, 1))
        expected_losses = self.huber_loss(y_true, y_pred)

        sample_weight = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3, 1))
        loss = h_obj(
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )
        actual_loss = np.multiply(expected_losses, sample_weight)
        actual_loss = np.sum(actual_loss) / self.batch_size
        self.assertAlmostEqual(loss, actual_loss, 3)

    def test_zero_weighted(self):
        self.setup()
        h_obj = losses.Huber()
        sample_weight = 0
        loss = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_non_default_delta(self):
        self.setup(delta=0.8)
        h_obj = losses.Huber(delta=0.8)
        sample_weight = 2.3
        loss = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        actual_loss = (
            sample_weight * np.sum(self.expected_losses) / self.batch_size
        )
        self.assertAlmostEqual(loss, actual_loss, 3)

    def test_loss_with_non_default_dtype(self):
        # Test case for GitHub issue:
        # https://github.com/tensorflow/tensorflow/issues/39004
        # TODO
        pass


class LogCoshTest(testing.TestCase):
    def setup(self):
        y_true = np.asarray([[1, 9, 2], [-5, -2, 6]], dtype=np.float32)
        y_pred = np.asarray([[4, 8, 12], [8, 1, 3]], dtype=np.float32)

        self.batch_size = 6
        error = y_pred - y_true
        self.expected_losses = np.log((np.exp(error) + np.exp(-error)) / 2)

        self.y_true = y_true
        self.y_pred = y_pred

    def test_config(self):
        logcosh_obj = losses.LogCosh(reduction="sum", name="logcosh_loss")
        self.assertEqual(logcosh_obj.name, "logcosh_loss")
        self.assertEqual(logcosh_obj.reduction, "sum")

    def test_unweighted(self):
        self.setup()
        logcosh_obj = losses.LogCosh()

        loss = logcosh_obj(self.y_true, self.y_pred)
        expected_loss = np.sum(self.expected_losses) / self.batch_size
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_scalar_weighted(self):
        self.setup()
        logcosh_obj = losses.LogCosh()
        sample_weight = 2.3

        loss = logcosh_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )
        expected_loss = (
            sample_weight * np.sum(self.expected_losses) / self.batch_size
        )
        self.assertAlmostEqual(loss, expected_loss, 3)

        # Verify we get the same output when the same input is given
        loss_2 = logcosh_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )
        self.assertAlmostEqual(loss, loss_2, 3)

    def test_sample_weighted(self):
        self.setup()
        logcosh_obj = losses.LogCosh()

        sample_weight = np.asarray([1.2, 3.4])
        loss = logcosh_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )

        expected_loss = np.multiply(
            self.expected_losses,
            np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3)),
        )
        expected_loss = np.sum(expected_loss) / self.batch_size
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_timestep_weighted(self):
        self.setup()
        logcosh_obj = losses.LogCosh()
        y_true = np.asarray([1, 9, 2, -5, -2, 6]).reshape(2, 3, 1)
        y_pred = np.asarray([4, 8, 12, 8, 1, 3]).reshape(2, 3, 1)
        error = y_pred - y_true
        expected_losses = np.log((np.exp(error) + np.exp(-error)) / 2)
        sample_weight = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3, 1))

        loss = logcosh_obj(
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )
        expected_loss = (
            np.sum(expected_losses * sample_weight) / self.batch_size
        )
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_zero_weighted(self):
        self.setup()
        logcosh_obj = losses.LogCosh()
        sample_weight = 0
        loss = logcosh_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )
        self.assertAlmostEqual(loss, 0.0, 3)


class KLDivergenceTest(testing.TestCase):
    def setup(self):
        self.y_pred = np.asarray(
            [0.4, 0.9, 0.12, 0.36, 0.3, 0.4], dtype=np.float32
        ).reshape((2, 3))
        self.y_true = np.asarray(
            [0.5, 0.8, 0.12, 0.7, 0.43, 0.8], dtype=np.float32
        ).reshape((2, 3))

        self.batch_size = 2
        self.expected_losses = np.multiply(
            self.y_true, np.log(self.y_true / self.y_pred)
        )

    def test_config(self):
        k_obj = losses.KLDivergence(reduction="sum", name="kld")
        self.assertEqual(k_obj.name, "kld")
        self.assertEqual(k_obj.reduction, "sum")

    def test_unweighted(self):
        self.setup()
        k_obj = losses.KLDivergence()

        loss = k_obj(self.y_true, self.y_pred)
        expected_loss = np.sum(self.expected_losses) / self.batch_size
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_scalar_weighted(self):
        self.setup()
        k_obj = losses.KLDivergence()
        sample_weight = 2.3

        loss = k_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        expected_loss = (
            sample_weight * np.sum(self.expected_losses) / self.batch_size
        )
        self.assertAlmostEqual(loss, expected_loss, 3)

        # Verify we get the same output when the same input is given
        loss_2 = k_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, loss_2, 3)

    def test_sample_weighted(self):
        self.setup()
        k_obj = losses.KLDivergence()
        sample_weight = np.asarray([1.2, 3.4], dtype=np.float32).reshape((2, 1))
        loss = k_obj(self.y_true, self.y_pred, sample_weight=sample_weight)

        expected_loss = np.multiply(
            self.expected_losses,
            np.asarray(
                [1.2, 1.2, 1.2, 3.4, 3.4, 3.4], dtype=np.float32
            ).reshape(2, 3),
        )
        expected_loss = np.sum(expected_loss) / self.batch_size
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_timestep_weighted(self):
        self.setup()
        k_obj = losses.KLDivergence()
        y_true = self.y_true.reshape(2, 3, 1)
        y_pred = self.y_pred.reshape(2, 3, 1)
        sample_weight = np.asarray([3, 6, 5, 0, 4, 2]).reshape(2, 3)
        expected_losses = np.sum(
            np.multiply(y_true, np.log(y_true / y_pred)), axis=-1
        )
        loss = k_obj(y_true, y_pred, sample_weight=sample_weight)

        num_timesteps = 3
        expected_loss = np.sum(expected_losses * sample_weight) / (
            self.batch_size * num_timesteps
        )
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_zero_weighted(self):
        self.setup()
        k_obj = losses.KLDivergence()
        loss = k_obj(self.y_true, self.y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0, 3)


class PoissonTest(testing.TestCase):
    def setup(self):
        self.y_pred = np.asarray([1, 9, 2, 5, 2, 6], dtype=np.float32).reshape(
            (2, 3)
        )
        self.y_true = np.asarray([4, 8, 12, 8, 1, 3], dtype=np.float32).reshape(
            (2, 3)
        )

        self.batch_size = 6
        self.expected_losses = self.y_pred - np.multiply(
            self.y_true, np.log(self.y_pred)
        )

    def test_config(self):
        poisson_obj = losses.Poisson(reduction="sum", name="poisson")
        self.assertEqual(poisson_obj.name, "poisson")
        self.assertEqual(poisson_obj.reduction, "sum")

    def test_unweighted(self):
        self.setup()
        poisson_obj = losses.Poisson()

        loss = poisson_obj(self.y_true, self.y_pred)
        expected_loss = np.sum(self.expected_losses) / self.batch_size
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_scalar_weighted(self):
        self.setup()
        poisson_obj = losses.Poisson()
        sample_weight = 2.3
        loss = poisson_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )
        expected_loss = (
            sample_weight * np.sum(self.expected_losses) / self.batch_size
        )
        self.assertAlmostEqual(loss, expected_loss, 3)
        self.assertAlmostEqual(loss, expected_loss, 3)

        # Verify we get the same output when the same input is given
        loss_2 = poisson_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )
        self.assertAlmostEqual(loss, loss_2, 3)

    def test_sample_weighted(self):
        self.setup()
        poisson_obj = losses.Poisson()

        sample_weight = np.asarray([1.2, 3.4]).reshape((2, 1))
        loss = poisson_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )

        expected_loss = np.multiply(
            self.expected_losses,
            np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3)),
        )
        expected_loss = np.sum(expected_loss) / self.batch_size
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_timestep_weighted(self):
        self.setup()
        poisson_obj = losses.Poisson()
        y_true = self.y_true.reshape(2, 3, 1)
        y_pred = self.y_pred.reshape(2, 3, 1)
        sample_weight = np.asarray([3, 6, 5, 0, 4, 2]).reshape(2, 3, 1)
        expected_losses = y_pred - np.multiply(y_true, np.log(y_pred))

        loss = poisson_obj(
            y_true,
            y_pred,
            sample_weight=np.asarray(sample_weight).reshape((2, 3)),
        )
        expected_loss = (
            np.sum(expected_losses * sample_weight) / self.batch_size
        )
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_zero_weighted(self):
        self.setup()
        poisson_obj = losses.Poisson()
        loss = poisson_obj(self.y_true, self.y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0, 3)


class BinaryCrossentropyTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.BinaryCrossentropy(name="bce", axis=-1)
        )

    def test_all_correct_unweighted(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")
        bce_obj = losses.BinaryCrossentropy()
        loss = bce_obj(y_true, y_true)
        self.assertAlmostEqual(loss, 0.0)

        # Test with logits.
        logits = np.array(
            [
                [10.0, -10.0, -10.0],
                [-10.0, 10.0, -10.0],
                [-10.0, -10.0, 10.0],
            ]
        )
        bce_obj = losses.BinaryCrossentropy(from_logits=True)
        loss = bce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.0)

    def test_unweighted(self):
        y_true = np.array([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.array([1, 1, 1, 0], dtype=np.float32).reshape([2, 2])
        bce_obj = losses.BinaryCrossentropy()
        loss = bce_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 3.98559)

        # Test with logits.
        y_true = np.array([[1, 0, 1], [0, 1, 1]])
        logits = np.array([[10.0, -10.0, 10.0], [10.0, 10.0, -10.0]])
        bce_obj = losses.BinaryCrossentropy(from_logits=True)
        loss = bce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 3.3333)

    def test_scalar_weighted(self):
        bce_obj = losses.BinaryCrossentropy()
        y_true = np.array([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.array([1, 1, 1, 0], dtype="float32").reshape([2, 2])
        loss = bce_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 9.1668)

        # Test with logits.
        y_true = np.array([[1, 0, 1], [0, 1, 1]])
        logits = np.array([[10.0, -10.0, 10.0], [10.0, 10.0, -10.0]])
        bce_obj = losses.BinaryCrossentropy(from_logits=True)
        loss = bce_obj(y_true, logits, sample_weight=2.3)
        self.assertAlmostEqual(loss, 7.666)

    def test_sample_weighted(self):
        bce_obj = losses.BinaryCrossentropy()
        y_true = np.array([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.array([1, 1, 1, 0], dtype="float32").reshape([2, 2])
        sample_weight = np.array([1.2, 3.4]).reshape((2, 1))
        loss = bce_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 4.7827)

        # Test with logits.
        y_true = np.array([[1, 0, 1], [0, 1, 1]])
        logits = np.array([[10.0, -10.0, 10.0], [10.0, 10.0, -10.0]])
        weights = np.array([4, 3])
        bce_obj = losses.BinaryCrossentropy(from_logits=True)
        loss = bce_obj(y_true, logits, sample_weight=weights)
        self.assertAlmostEqual(loss, 10.0)

    def test_no_reduction(self):
        y_true = np.array([[1, 0, 1], [0, 1, 1]])
        logits = np.array([[10.0, -10.0, 10.0], [10.0, 10.0, -10.0]])
        bce_obj = losses.BinaryCrossentropy(from_logits=True, reduction=None)
        loss = bce_obj(y_true, logits)
        self.assertAllClose(loss, [0.0, 6.666], atol=1e-3)

    def test_label_smoothing(self):
        logits = np.array([[10.0, -10.0, -10.0]])
        y_true = np.array([[1, 0, 1]])
        label_smoothing = 0.1
        bce_obj = losses.BinaryCrossentropy(
            from_logits=True, label_smoothing=label_smoothing
        )
        loss = bce_obj(y_true, logits)
        expected_value = (10.0 + 5.0 * label_smoothing) / 3.0
        self.assertAlmostEqual(loss, expected_value)

    def test_shape_mismatch(self):
        y_true = np.array([[0], [1], [2]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]]
        )
        cce_obj = losses.BinaryCrossentropy()
        with self.assertRaisesRegex(ValueError, "must have the same shape"):
            cce_obj(y_true, y_pred)


class CategoricalCrossentropyTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.CategoricalCrossentropy(name="cce", axis=-1)
        )

    def test_all_correct_unweighted(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="int64")
        y_pred = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype="float32",
        )
        cce_obj = losses.CategoricalCrossentropy()
        loss = cce_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.0)

        # Test with logits.
        logits = np.array(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        )
        cce_obj = losses.CategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.0)

    def test_unweighted(self):
        cce_obj = losses.CategoricalCrossentropy()
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        loss = cce_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.3239)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.0573)

    def test_scalar_weighted(self):
        cce_obj = losses.CategoricalCrossentropy()
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        loss = cce_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 0.7449)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits, sample_weight=2.3)
        self.assertAlmostEqual(loss, 0.1317)

    def test_sample_weighted(self):
        cce_obj = losses.CategoricalCrossentropy()
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        sample_weight = np.array([[1.2], [3.4], [5.6]]).reshape((3, 1))
        loss = cce_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 1.0696)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.31829)

    def test_no_reduction(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalCrossentropy(
            from_logits=True, reduction=None
        )
        loss = cce_obj(y_true, logits)
        self.assertAllClose((0.001822, 0.000459, 0.169846), loss, 3)

    def test_label_smoothing(self):
        logits = np.array([[100.0, -100.0, -100.0]])
        y_true = np.array([[1, 0, 0]])
        label_smoothing = 0.1
        cce_obj = losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=label_smoothing
        )
        loss = cce_obj(y_true, logits)
        expected_value = 400.0 * label_smoothing / 3.0
        self.assertAlmostEqual(loss, expected_value)

    def test_label_smoothing_ndarray(self):
        logits = np.asarray([[100.0, -100.0, -100.0]])
        y_true = np.asarray([[1, 0, 0]])
        label_smoothing = 0.1
        cce_obj = losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=label_smoothing
        )
        loss = cce_obj(y_true, logits)
        expected_value = 400.0 * label_smoothing / 3.0
        self.assertAlmostEqual(loss, expected_value)

    def test_shape_mismatch(self):
        y_true = np.array([[0], [1], [2]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]]
        )

        cce_obj = losses.CategoricalCrossentropy()
        with self.assertRaisesRegex(ValueError, "must have the same shape"):
            cce_obj(y_true, y_pred)


class SparseCategoricalCrossentropyTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.SparseCategoricalCrossentropy(name="scce")
        )

    def test_all_correct_unweighted(self):
        y_true = np.array([[0], [1], [2]], dtype="int64")
        y_pred = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype="float32",
        )
        cce_obj = losses.SparseCategoricalCrossentropy()
        loss = cce_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.0, 3)

        # Test with logits.
        logits = np.array(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        )
        cce_obj = losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_unweighted(self):
        cce_obj = losses.SparseCategoricalCrossentropy()
        y_true = np.array([0, 1, 2])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        loss = cce_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.3239, 3)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.0573, 3)

    def test_scalar_weighted(self):
        cce_obj = losses.SparseCategoricalCrossentropy()
        y_true = np.array([[0], [1], [2]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        loss = cce_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 0.7449, 3)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits, sample_weight=2.3)
        self.assertAlmostEqual(loss, 0.1317, 3)

    def test_sample_weighted(self):
        cce_obj = losses.SparseCategoricalCrossentropy()
        y_true = np.array([[0], [1], [2]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        sample_weight = np.array([[1.2], [3.4], [5.6]]).reshape((3, 1))
        loss = cce_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 1.0696, 3)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.31829, 3)

    def test_no_reduction(self):
        y_true = np.array([[0], [1], [2]])
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=None
        )
        loss = cce_obj(y_true, logits)
        self.assertAllClose((0.001822, 0.000459, 0.169846), loss, 3)


class BinaryFocalCrossentropyTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.BinaryFocalCrossentropy(name="bfce")
        )

    def test_all_correct_unweighted(self):
        y_true = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype="float32",
        )
        obj = losses.BinaryFocalCrossentropy(gamma=1.5)
        loss = obj(y_true, y_true)
        self.assertAlmostEqual(loss, 0.0, 3)

        # Test with logits.
        logits = np.array(
            [
                [100.0, -100.0, -100.0],
                [-100.0, 100.0, -100.0],
                [-100.0, -100.0, 100.0],
            ]
        )
        obj = losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=True)
        loss = obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_unweighted(self):
        y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.asarray([0.9, 0.8, 0.7, 0.2], dtype=np.float32).reshape(
            [2, 2]
        )
        obj = losses.BinaryFocalCrossentropy(gamma=2.0)
        loss = obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.268, 3)

        # Test with logits.
        y_true = np.array([[1, 1, 0], [0, 1, 0]], dtype="float32")
        logits = np.array([[1.5, -2.7, 2.9], [-3.8, 1.2, -4.5]])
        obj = losses.BinaryFocalCrossentropy(gamma=3.0, from_logits=True)
        loss = obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.799, 3)

    def test_scalar_weighted(self):
        y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.asarray([0.9, 0.8, 0.7, 0.2], dtype=np.float32).reshape(
            [2, 2]
        )
        obj = losses.BinaryFocalCrossentropy(gamma=2.0)
        loss = obj(y_true, y_pred, sample_weight=1.23)
        self.assertAlmostEqual(loss, 0.3296, 3)

        # Test with logits.
        y_true = np.array([[1, 1, 0], [0, 1, 0]], dtype="float32")
        logits = np.array([[1.5, -2.7, 2.9], [-3.8, 1.2, -4.5]])
        obj = losses.BinaryFocalCrossentropy(gamma=3.0, from_logits=True)
        loss = obj(y_true, logits, sample_weight=3.21)
        self.assertAlmostEqual(loss, 2.565, 3)

    def test_sample_weighted(self):
        y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.asarray([0.9, 0.8, 0.7, 0.2], dtype=np.float32).reshape(
            [2, 2]
        )
        sample_weight = np.array([1.2, 3.4]).reshape((2, 1))
        obj = losses.BinaryFocalCrossentropy(gamma=2.0)
        loss = obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.34415, 3)

        # Test with logits.
        y_true = np.array([[1, 1, 0], [0, 1, 0]], dtype="float32")
        logits = np.array([[1.5, -2.7, 2.9], [-3.8, 1.2, -4.5]])
        obj = losses.BinaryFocalCrossentropy(gamma=3.0, from_logits=True)
        loss = obj(y_true, logits, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.95977, 3)

    def test_no_reduction(self):
        y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.asarray([0.9, 0.8, 0.7, 0.2], dtype=np.float32).reshape(
            [2, 2]
        )
        obj = losses.BinaryFocalCrossentropy(
            gamma=2.0,
            reduction=None,
        )
        loss = obj(y_true, y_pred)
        self.assertAllClose(loss, (0.5155, 0.0205), 3)


class CategoricalFocalCrossentropyTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.CategoricalFocalCrossentropy(name="cfce")
        )

    def test_all_correct_unweighted(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="int64")
        y_pred = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype="float32",
        )
        cce_obj = losses.CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0)
        loss = cce_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.0, 3)

        # Test with logits.
        logits = np.array(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        )
        cce_obj = losses.CategoricalFocalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_unweighted(self):
        cce_obj = losses.CategoricalFocalCrossentropy()
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        loss = cce_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.02059, 3)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalFocalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.000345, 3)

    def test_scalar_weighted(self):
        cce_obj = losses.CategoricalFocalCrossentropy()
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        loss = cce_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 0.047368, 3)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalFocalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits, sample_weight=2.3)
        self.assertAlmostEqual(loss, 0.000794, 4)

    def test_sample_weighted(self):
        cce_obj = losses.CategoricalFocalCrossentropy()
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        sample_weight = np.array([[1.2], [3.4], [5.6]]).reshape((3, 1))
        loss = cce_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.06987, 3)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalFocalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.001933, 3)

    def test_no_reduction(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalFocalCrossentropy(
            from_logits=True, reduction=None
        )
        loss = cce_obj(y_true, logits)
        self.assertAllClose(
            (1.5096224e-09, 2.4136547e-11, 1.0360638e-03),
            loss,
            3,
        )

    def test_label_smoothing(self):
        logits = np.array([[4.9, -0.5, 2.05]])
        y_true = np.array([[1, 0, 0]])
        label_smoothing = 0.1

        cce_obj = losses.CategoricalFocalCrossentropy(
            from_logits=True, label_smoothing=label_smoothing
        )
        loss = cce_obj(y_true, logits)

        expected_value = 0.06685
        self.assertAlmostEqual(loss, expected_value, 3)
