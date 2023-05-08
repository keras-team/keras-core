from keras_core import callbacks
from keras_core import optimizers
from keras_core import testing
from keras_core.testing import test_utils
from keras_core.utils import io_utils
from keras_core.utils import numerical_utils


class LearningRateSchedulerTest(testing.TestCase):
    def setUp(self):
        (x_train, y_train), (x_test, y_test) = test_utils.get_test_data(
            train_samples=10,
            test_samples=10,
            input_shape=(3,),
            num_classes=2,
        )
        y_test = numerical_utils.to_categorical(y_test)
        y_train = numerical_utils.to_categorical(y_train)

        model = test_utils.get_small_sequential_mlp(
            num_hidden=5,
            num_classes=2,
        )
        model.compile(
            loss="mse",
            optimizer="sgd",
        )

        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def test_updates_learning_rate(self):
        lr_scheduler = callbacks.LearningRateScheduler(
            lambda step: 1.0 / (2.0 + step), verbose=1
        )

        self.model.fit(
            self.x_train,
            self.y_train,
            callbacks=[lr_scheduler],
            epochs=1,
        )

        self.assertEqual(self.model.optimizer.learning_rate.value, 0.5)

    def test_verbose_logging(self):
        lr_scheduler = callbacks.LearningRateScheduler(
            lambda step: 1.0 / (1.0 + step), verbose=1
        )
        io_utils.disable_interactive_logging()

        with self.assertLogs(level="INFO") as logs:
            self.model.fit(
                self.x_train,
                self.y_train,
                callbacks=[lr_scheduler],
                epochs=1,
            )
            expected_log = "LearningRateScheduler setting learning rate to 1.0"
            self.assertTrue(any(expected_log in log for log in logs.output))

    def test_schedule_dependent_on_previous_learning_rate(self):
        lr_scheduler = callbacks.LearningRateScheduler(lambda step, lr: lr / 2)

        initial_lr = 0.03
        self.model.compile(
            loss="mse",
            optimizer=optimizers.Adam(initial_lr),
        )

        self.model.fit(
            self.x_train,
            self.y_train,
            callbacks=[lr_scheduler],
            epochs=2,
        )
        self.assertEqual(
            self.model.optimizer.learning_rate.value, initial_lr / 4.0
        )

    def test_throws_when_optimizer_has_schedule(self):
        lr_scheduler = callbacks.LearningRateScheduler(lambda step, lr: lr / 2)

        self.model.compile(
            loss="mse",
            optimizer=optimizers.Adam(
                optimizers.schedules.PolynomialDecay(
                    initial_learning_rate=0.1, decay_steps=10
                )
            ),
        )

        with self.assertRaisesRegex(
            TypeError,
            "This optimizer was created with a `LearningRateSchedule`",
        ):
            self.model.fit(
                self.x_train,
                self.y_train,
                callbacks=[lr_scheduler],
                epochs=2,
            )
