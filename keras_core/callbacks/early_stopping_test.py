import numpy as np

from keras_core import callbacks
from keras_core import layers
from keras_core import models
from keras_core import testing
from keras_core.testing import test_utils
from keras_core.utils import np_utils

TRAIN_SAMPLES = 10
TEST_SAMPLES = 10
NUM_CLASSES = 2
INPUT_DIM = 3
BATCH_SIZE = 5


class EarlyStoppingTest(testing.TestCase):
    def test_early_stopping(self):
        (x_train, y_train), (
            x_test,
            y_test,
        ) = test_utils.get_test_data(
            train_samples=TRAIN_SAMPLES,
            test_samples=TEST_SAMPLES,
            input_shape=(INPUT_DIM,),
            num_classes=NUM_CLASSES,
        )
        y_test = np_utils.to_categorical(y_test)
        y_train = np_utils.to_categorical(y_train)
        model = models.Sequential(
            (
                layers.Dense(1, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            )
        )
        model.compile(
            loss="mae",
            optimizer="adam",
            metrics=["mse"],
        )

        cases = [
            ("max", "val_mse"),
            ("min", "val_loss"),
            ("auto", "val_mse"),
            ("auto", "loss"),
            ("unknown", "unknown"),
        ]
        for mode, monitor in cases:
            patience = 0
            cbks = [
                callbacks.EarlyStopping(
                    patience=patience, monitor=monitor, mode=mode
                )
            ]
            model.fit(
                x_train,
                y_train,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                callbacks=cbks,
                epochs=5,
                verbose=0,
            )

    def test_early_stopping_patience(self):
        cases = [0, 1, 2, 3]
        losses = [10.0, 9.0, 8.0, 9.0, 8.9, 8.8, 8.7, 8.6, 8.5]

        for patience in cases:
            stopper = callbacks.EarlyStopping(monitor="loss", patience=patience)
            stopper.model = models.Sequential()
            stopper.model.compile(loss="mse", optimizer="sgd")
            stopper.on_train_begin()

            for epoch, loss in enumerate(losses):
                stopper.on_epoch_end(epoch=epoch, logs={"loss": loss})
                if stopper.model.stop_training:
                    break

            self.assertEqual(stopper.stopped_epoch, max(patience, 1) + 2)

    def test_early_stopping_reuse(self):
        patience = 3
        data = np.random.random((100, 1))
        labels = np.where(data > 0.5, 1, 0)
        model = models.Sequential(
            (
                layers.Dense(1, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            )
        )
        model.compile(
            optimizer="sgd",
            loss="mae",
            metrics=["mse"],
        )
        weights = model.get_weights()

        # This should allow training to go for at least `patience` epochs
        model.set_weights(weights)

        stopper = callbacks.EarlyStopping(monitor="mse", patience=patience)
        hist = model.fit(
            data, labels, callbacks=[stopper], verbose=0, epochs=20
        )
        assert len(hist.epoch) >= patience

    def test_early_stopping_with_baseline(self):
        baseline = 0.6
        (data, labels), _ = test_utils.get_test_data(
            train_samples=100,
            test_samples=50,
            input_shape=(1,),
            num_classes=NUM_CLASSES,
        )
        model = models.Sequential(
            (
                layers.Dense(1, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            )
        )
        model.compile(optimizer="sgd", loss="mae", metrics=["mse"])

        patience = 3
        stopper = callbacks.EarlyStopping(
            monitor="mse", patience=patience, baseline=baseline
        )
        hist = model.fit(
            data, labels, callbacks=[stopper], verbose=0, epochs=20
        )
        assert len(hist.epoch) >= patience

    def test_early_stopping_final_weights_when_restoring_model_weights(self):
        class DummyModel:
            def __init__(self):
                self.stop_training = False
                self.weights = -1

            def get_weights(self):
                return self.weights

            def set_weights(self, weights):
                self.weights = weights

            def set_weight_to_epoch(self, epoch):
                self.weights = epoch

        early_stop = callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
        early_stop.model = DummyModel()
        losses = [0.2, 0.15, 0.1, 0.11, 0.12]
        # The best configuration is in the epoch 2 (loss = 0.1000).
        epochs_trained = 0
        early_stop.on_train_begin()
        for epoch in range(len(losses)):
            epochs_trained += 1
            early_stop.model.set_weight_to_epoch(epoch=epoch)
            early_stop.on_epoch_end(epoch, logs={"val_loss": losses[epoch]})
            if early_stop.model.stop_training:
                break
        # The best configuration is in epoch 2 (loss = 0.1000),
        # and while patience = 2, we're restoring the best weights,
        # so we end up at the epoch with the best weights, i.e. epoch 2
        self.assertEqual(early_stop.model.get_weights(), 2)

        # Check early stopping when no model beats the baseline.
        early_stop = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            baseline=0.5,
            restore_best_weights=True,
        )
        early_stop.model = DummyModel()
        losses = [0.9, 0.8, 0.7, 0.71, 0.72, 0.73]
        # The best configuration is in the epoch 2 (loss = 0.7000).
        epochs_trained = 0
        early_stop.on_train_begin()
        for epoch in range(len(losses)):
            epochs_trained += 1
            early_stop.model.set_weight_to_epoch(epoch=epoch)
            early_stop.on_epoch_end(epoch, logs={"val_loss": losses[epoch]})
            if early_stop.model.stop_training:
                break
        # No epoch improves on the baseline, so we should train for only 5
        # epochs, and restore the second model.
        self.assertEqual(epochs_trained, 5)
        self.assertEqual(early_stop.model.get_weights(), 2)

    def test_early_stopping_with_start_from_epoch(self):
        (data, labels), _ = test_utils.get_test_data(
            train_samples=TRAIN_SAMPLES,
            test_samples=TEST_SAMPLES,
            input_shape=(INPUT_DIM,),
            num_classes=NUM_CLASSES,
        )
        labels = np_utils.to_categorical(labels)
        model = models.Sequential(
            (
                layers.Dense(1, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            )
        )
        model.compile(optimizer="sgd", loss="mae", metrics=["mse"])
        start_from_epoch = 2
        patience = 3
        stopper = callbacks.EarlyStopping(
            monitor="mse",
            patience=patience,
            start_from_epoch=start_from_epoch,
        )
        history = model.fit(
            data, labels, callbacks=[stopper], verbose=0, epochs=20
        )
        # Test 'patience' argument functions correctly when used
        # in conjunction with 'start_from_epoch'.
        self.assertGreaterEqual(len(history.epoch), patience + start_from_epoch)

        start_from_epoch = 2
        patience = 0
        stopper = callbacks.EarlyStopping(
            monitor="mse",
            patience=patience,
            start_from_epoch=start_from_epoch,
        )
        history = model.fit(
            data, labels, callbacks=[stopper], verbose=0, epochs=20
        )
        # Test for boundary condition when 'patience' = 0.
        self.assertGreaterEqual(len(history.epoch), start_from_epoch)
