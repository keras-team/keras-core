import numpy as np
import pytest
from absl import logging

from keras_core import callbacks
from keras_core import layers
from keras_core import losses
from keras_core import optimizers
from keras_core import testing
from keras_core.models.sequential import Sequential


class LambdaCallbackTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_LambdaCallback(self):
        """Test standard LambdaCallback functionalities with training."""
        BATCH_SIZE = 4
        model = Sequential(
            [layers.Input(shape=(2,), batch_size=BATCH_SIZE), layers.Dense(1)]
        )
        model.compile(
            optimizer=optimizers.SGD(), loss=losses.MeanSquaredError()
        )
        x = np.random.randn(16, 2)
        y = np.random.randn(16, 1)
        lambda_log_callback = callbacks.LambdaCallback(
            on_train_begin=lambda logs: logging.warning("on_train_begin"),
            on_epoch_begin=lambda epoch, logs: logging.warning(
                "on_epoch_begin"
            ),
            on_epoch_end=lambda epoch, logs: logging.warning("on_epoch_end"),
            on_train_end=lambda logs: logging.warning("on_train_end"),
        )
        with self.assertLogs(level="WARNING") as logs:
            model.fit(
                x,
                y,
                batch_size=BATCH_SIZE,
                validation_split=0.2,
                callbacks=[lambda_log_callback],
                epochs=5,
                verbose=0,
            )
            self.assertTrue
            (any("on_train_begin" in log for log in logs.output))
            self.assertTrue
            (any("on_epoch_begin" in log for log in logs.output))
            self.assertTrue
            (any("on_epoch_end" in log for log in logs.output))
            self.assertTrue
            (any("on_train_end" in log for log in logs.output))

    @pytest.mark.requires_trainable_backend
    def test_LambdaCallback_with_batches(self):
        """Test LambdaCallback's behavior with batch-level callbacks."""
        BATCH_SIZE = 4
        model = Sequential(
            [layers.Input(shape=(2,), batch_size=BATCH_SIZE), layers.Dense(1)]
        )
        model.compile(
            optimizer=optimizers.SGD(), loss=losses.MeanSquaredError()
        )
        x = np.random.randn(16, 2)
        y = np.random.randn(16, 1)
        lambda_log_callback = callbacks.LambdaCallback(
            on_train_batch_begin=lambda batch, logs: logging.warning(
                "on_train_batch_begin"
            ),
            on_train_batch_end=lambda batch, logs: logging.warning(
                "on_train_batch_end"
            ),
        )
        with self.assertLogs(level="WARNING") as logs:
            model.fit(
                x,
                y,
                batch_size=BATCH_SIZE,
                validation_split=0.2,
                callbacks=[lambda_log_callback],
                epochs=5,
                verbose=0,
            )
            self.assertTrue(
                any("on_train_batch_begin" in log for log in logs.output)
            )
            self.assertTrue(
                any("on_train_batch_end" in log for log in logs.output)
            )

    @pytest.mark.requires_trainable_backend
    def test_LambdaCallback_with_kwargs(self):
        """Test LambdaCallback's behavior with custom defined callback."""
        BATCH_SIZE = 4
        model = Sequential(
            [layers.Input(shape=(2,), batch_size=BATCH_SIZE), layers.Dense(1)]
        )
        model.compile(
            optimizer=optimizers.SGD(), loss=losses.MeanSquaredError()
        )
        x = np.random.randn(16, 2)
        y = np.random.randn(16, 1)
        model.fit(
            x, y, batch_size=BATCH_SIZE, epochs=1, verbose=0
        )  # Train briefly for evaluation to work.

        def custom_on_test_begin(logs):
            logging.warning("custom_on_test_begin_executed")

        lambda_log_callback = callbacks.LambdaCallback(
            on_test_begin=custom_on_test_begin
        )
        with self.assertLogs(level="WARNING") as logs:
            model.evaluate(
                x,
                y,
                batch_size=BATCH_SIZE,
                callbacks=[lambda_log_callback],
                verbose=0,
            )
            self.assertTrue(
                any(
                    "custom_on_test_begin_executed" in log
                    for log in logs.output
                )
            )

    @pytest.mark.requires_trainable_backend
    def test_LambdaCallback_no_args(self):
        """Test initializing LambdaCallback without any arguments."""
        lambda_callback = callbacks.LambdaCallback()
        self.assertIsInstance(lambda_callback, callbacks.LambdaCallback)

    @pytest.mark.requires_trainable_backend
    def test_LambdaCallback_with_additional_kwargs(self):
        """Test initializing LambdaCallback with non-predefined kwargs."""

        def custom_callback(logs):
            pass

        lambda_callback = callbacks.LambdaCallback(
            custom_method=custom_callback
        )
        self.assertTrue(hasattr(lambda_callback, "custom_method"))

    @pytest.mark.requires_trainable_backend
    def test_LambdaCallback_during_prediction(self):
        """Test LambdaCallback's functionality during model prediction."""
        BATCH_SIZE = 4
        model = Sequential(
            [layers.Input(shape=(2,), batch_size=BATCH_SIZE), layers.Dense(1)]
        )
        model.compile(
            optimizer=optimizers.SGD(), loss=losses.MeanSquaredError()
        )
        x = np.random.randn(16, 2)

        def custom_on_predict_begin(logs):
            logging.warning("on_predict_begin_executed")

        lambda_callback = callbacks.LambdaCallback(
            on_predict_begin=custom_on_predict_begin
        )
        with self.assertLogs(level="WARNING") as logs:
            model.predict(
                x, batch_size=BATCH_SIZE, callbacks=[lambda_callback], verbose=0
            )
            self.assertTrue(
                any("on_predict_begin_executed" in log for log in logs.output)
            )
