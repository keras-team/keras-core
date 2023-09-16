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
            self.assertTrue(any("on_train_begin" in log for log in logs.output))
            self.assertTrue(any("on_epoch_begin" in log for log in logs.output))
            self.assertTrue(any("on_epoch_end" in log for log in logs.output))
            self.assertTrue(any("on_train_end" in log for log in logs.output))

    @pytest.mark.requires_trainable_backend
    def test_LambdaCallback_with_batches(self):
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

        custom_on_test_begin = lambda logs: logging.warning(
            "custom_on_test_begin_executed"
        )
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
