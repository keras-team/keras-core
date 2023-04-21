import tensorflow as tf


class FalsePositivesTest(testing.TestCase):
    def test_config(self):
        fp_obj = metrics.FalsePositives(name="my_fp", thresholds=[0.4, 0.9])
        self.assertEqual(fp_obj.name, "my_fp")
        self.assertLen(fp_obj.variables, 1)
        self.assertEqual(fp_obj.thresholds, [0.4, 0.9])

        # Check save and restore config
        fp_obj2 = metrics.FalsePositives.from_config(fp_obj.get_config())
        self.assertEqual(fp_obj2.name, "my_fp")
        self.assertLen(fp_obj2.variables, 1)
        self.assertEqual(fp_obj2.thresholds, [0.4, 0.9])

    def test_unweighted(self):
        fp_obj = metrics.FalsePositives()
        self.evaluate(tf.compat.v1.variables_initializer(fp_obj.variables))

        y_true = tf.constant(
            ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        )
        y_pred = tf.constant(
            ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        )

        update_op = fp_obj.update_state(y_true, y_pred)
        self.evaluate(update_op)
        result = fp_obj.result()
        self.assertAllClose(7.0, result)

    def test_weighted(self):
        fp_obj = metrics.FalsePositives()
        self.evaluate(tf.compat.v1.variables_initializer(fp_obj.variables))
        y_true = tf.constant(
            ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        )
        y_pred = tf.constant(
            ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        )
        sample_weight = tf.constant((1.0, 1.5, 2.0, 2.5))
        result = fp_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(14.0, self.evaluate(result))

    def test_unweighted_with_thresholds(self):
        fp_obj = metrics.FalsePositives(thresholds=[0.15, 0.5, 0.85])
        self.evaluate(tf.compat.v1.variables_initializer(fp_obj.variables))

        y_pred = tf.constant(
            (
                (0.9, 0.2, 0.8, 0.1),
                (0.2, 0.9, 0.7, 0.6),
                (0.1, 0.2, 0.4, 0.3),
                (0, 1, 0.7, 0.3),
            )
        )
        y_true = tf.constant(
            ((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0), (1, 1, 1, 1))
        )

        update_op = fp_obj.update_state(y_true, y_pred)
        self.evaluate(update_op)
        result = fp_obj.result()
        self.assertAllClose([7.0, 4.0, 2.0], result)

    def test_weighted_with_thresholds(self):
        fp_obj = metrics.FalsePositives(thresholds=[0.15, 0.5, 0.85])
        self.evaluate(tf.compat.v1.variables_initializer(fp_obj.variables))

        y_pred = tf.constant(
            (
                (0.9, 0.2, 0.8, 0.1),
                (0.2, 0.9, 0.7, 0.6),
                (0.1, 0.2, 0.4, 0.3),
                (0, 1, 0.7, 0.3),
            )
        )
        y_true = tf.constant(
            ((0, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0), (1, 1, 1, 1))
        )
        sample_weight = (
            (1.0, 2.0, 3.0, 5.0),
            (7.0, 11.0, 13.0, 17.0),
            (19.0, 23.0, 29.0, 31.0),
            (5.0, 15.0, 10.0, 0),
        )

        result = fp_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose([125.0, 42.0, 12.0], self.evaluate(result))

    def test_threshold_limit(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Threshold values must be in \[0, 1\]. Received: \[-1, 2\]",
        ):
            metrics.FalsePositives(thresholds=[-1, 0.5, 2])

        with self.assertRaisesRegex(
            ValueError,
            r"Threshold values must be in \[0, 1\]. Received: \[None\]",
        ):
            metrics.FalsePositives(thresholds=[None])
