import os
import warnings

from keras_core import callbacks
from keras_core import layers
from keras_core import metrics
from keras_core import testing
from keras_core.models import Sequential
from keras_core.testing import test_utils
from keras_core.utils import numerical_utils

try:
    import h5py
except ImportError:
    h5py = None

TRAIN_SAMPLES = 10
TEST_SAMPLES = 10
NUM_CLASSES = 2
INPUT_DIM = 3
NUM_HIDDEN = 5
BATCH_SIZE = 5


class ModelCheckpointTest(testing.TestCase):
    def test_ModelCheckpoint(self):
        if h5py is None:
            return  # Skip test if models cannot be saved.

        model = Sequential(
            [
                layers.Dense(NUM_HIDDEN, activation="relu"),
                layers.Dense(NUM_CLASSES, activation="softmax"),
            ]
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer="sgd",
            metrics=[metrics.Accuracy("acc")],
        )

        temp_dir = self.get_temp_dir()

        # Save model to a subdir inside the temp_dir so we can test
        # automatic directory creation.
        filepath = os.path.join(temp_dir, "subdir", "checkpoint.keras")
        (x_train, y_train), (x_test, y_test) = test_utils.get_test_data(
            train_samples=TRAIN_SAMPLES,
            test_samples=TEST_SAMPLES,
            input_shape=(INPUT_DIM,),
            num_classes=NUM_CLASSES,
        )
        y_test = numerical_utils.to_categorical(y_test)
        y_train = numerical_utils.to_categorical(y_train)

        # Case 1
        monitor = "val_loss"
        save_best_only = False
        mode = "auto"

        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        self.assertTrue(os.path.exists(filepath))
        os.remove(filepath)

        # Case 2
        mode = "min"
        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        self.assertTrue(os.path.exists(filepath))
        os.remove(filepath)

        # Case 3
        mode = "max"
        monitor = "val_acc"
        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        self.assertTrue(os.path.exists(filepath))
        os.remove(filepath)

        # Case 4
        save_best_only = True
        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        self.assertTrue(os.path.exists(filepath))
        os.remove(filepath)

        # Case 5: metric not available.
        cbks = [
            callbacks.ModelCheckpoint(
                filepath, monitor="unknown", save_best_only=True
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        # File won't be written.
        self.assertFalse(os.path.exists(filepath))

        # Case 6
        with warnings.catch_warnings(record=True) as warning_logs:
            warnings.simplefilter("always")
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                mode="unknown",
            )
            self.assertIn(
                "ModelCheckpoint mode unknown is unknown",
                str(warning_logs[-1].message),
            )

        # Case 8a: `ModelCheckpoint` with an integer `save_freq`
        temp_dir = self.get_temp_dir()
        filepath = os.path.join(temp_dir, "checkpoint.epoch{epoch:02d}.keras")
        save_best_only = False
        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
                save_freq=15,
            )
        ]
        self.assertFalse(os.path.exists(filepath.format(epoch=3)))
        model.fit(
            x_train,
            y_train,
            batch_size=2,  # 5 batches / epoch, so should backup every 3 epochs
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=10,
            verbose=0,
        )
        self.assertFalse(os.path.exists(filepath.format(epoch=1)))
        self.assertFalse(os.path.exists(filepath.format(epoch=2)))
        self.assertTrue(os.path.exists(filepath.format(epoch=3)))
        self.assertFalse(os.path.exists(filepath.format(epoch=4)))
        self.assertFalse(os.path.exists(filepath.format(epoch=5)))
        self.assertTrue(os.path.exists(filepath.format(epoch=6)))
        self.assertFalse(os.path.exists(filepath.format(epoch=7)))
        self.assertFalse(os.path.exists(filepath.format(epoch=8)))
        self.assertTrue(os.path.exists(filepath.format(epoch=9)))
        os.remove(filepath.format(epoch=3))
        os.remove(filepath.format(epoch=6))
        os.remove(filepath.format(epoch=9))

        # Case 8b: `ModelCheckpoint` with int `save_freq` & `save_weights_only`
        temp_dir = self.get_temp_dir()
        filepath = os.path.join(
            temp_dir, "checkpoint.epoch{epoch:02d}.weights.h5"
        )
        cbks = [
            callbacks.ModelCheckpoint(
                filepath, monitor=monitor, save_freq=15, save_weights_only=True
            )
        ]
        self.assertFalse(os.path.exists(filepath.format(epoch=3)))
        model.fit(
            x_train,
            y_train,
            batch_size=2,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=10,
            verbose=0,
        )
        self.assertFalse(os.path.exists(filepath.format(epoch=1)))
        self.assertFalse(os.path.exists(filepath.format(epoch=2)))
        self.assertTrue(os.path.exists(filepath.format(epoch=3)))
        self.assertFalse(os.path.exists(filepath.format(epoch=4)))
        self.assertFalse(os.path.exists(filepath.format(epoch=5)))
        self.assertTrue(os.path.exists(filepath.format(epoch=6)))
        self.assertFalse(os.path.exists(filepath.format(epoch=7)))
        self.assertFalse(os.path.exists(filepath.format(epoch=8)))
        self.assertTrue(os.path.exists(filepath.format(epoch=9)))

        # Case 9: `ModelCheckpoint` with valid and invalid save_freq argument.
        with self.assertRaisesRegex(ValueError, "Unrecognized save_freq"):
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
                save_freq="invalid_save_freq",
            )
        # The following should not raise ValueError.
        callbacks.ModelCheckpoint(
            filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            mode=mode,
            save_freq="epoch",
        )
        callbacks.ModelCheckpoint(
            filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            mode=mode,
            save_freq=3,
        )

        # Case 10a: `ModelCheckpoint` save with batch in filename.
        temp_dir = self.get_temp_dir()
        filepath = os.path.join(
            temp_dir, "checkpoint.epoch{epoch:02d}batch{batch:02d}.keras"
        )
        cbks = [
            callbacks.ModelCheckpoint(filepath, monitor=monitor, save_freq=1)
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=5,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=5,
            verbose=1,
        )
        self.assertTrue(os.path.exists(filepath.format(epoch=1, batch=1)))
        self.assertTrue(os.path.exists(filepath.format(epoch=1, batch=2)))
        self.assertTrue(os.path.exists(filepath.format(epoch=2, batch=1)))
        self.assertTrue(os.path.exists(filepath.format(epoch=2, batch=2)))
        self.assertTrue(os.path.exists(filepath.format(epoch=3, batch=1)))
        self.assertTrue(os.path.exists(filepath.format(epoch=3, batch=2)))
        self.assertTrue(os.path.exists(filepath.format(epoch=4, batch=1)))
        self.assertTrue(os.path.exists(filepath.format(epoch=4, batch=2)))
        self.assertTrue(os.path.exists(filepath.format(epoch=5, batch=1)))
        self.assertTrue(os.path.exists(filepath.format(epoch=5, batch=2)))

        # Case 10b: `ModelCheckpoint` save weights with batch in filename.
        temp_dir = self.get_temp_dir()
        filepath = os.path.join(
            temp_dir, "checkpoint.epoch{epoch:02d}batch{batch:02d}.weights.h5"
        )
        cbks = [
            callbacks.ModelCheckpoint(
                filepath, monitor=monitor, save_freq=1, save_weights_only=True
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=5,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=5,
            verbose=1,
        )

        self.assertTrue(os.path.exists(filepath.format(epoch=1, batch=1)))
        self.assertTrue(os.path.exists(filepath.format(epoch=1, batch=2)))
        self.assertTrue(os.path.exists(filepath.format(epoch=2, batch=1)))
        self.assertTrue(os.path.exists(filepath.format(epoch=2, batch=2)))
        self.assertTrue(os.path.exists(filepath.format(epoch=3, batch=1)))
        self.assertTrue(os.path.exists(filepath.format(epoch=3, batch=2)))
        self.assertTrue(os.path.exists(filepath.format(epoch=4, batch=1)))
        self.assertTrue(os.path.exists(filepath.format(epoch=4, batch=2)))
        self.assertTrue(os.path.exists(filepath.format(epoch=5, batch=1)))
        self.assertTrue(os.path.exists(filepath.format(epoch=5, batch=2)))

        # Case 11: ModelCheckpoint saves model with initial_value_threshold
        # param
        mode = "max"
        monitor = "val_acc"
        initial_value_threshold = -0.01
        save_best_only = True
        filepath = os.path.join(temp_dir, "checkpoint.keras")
        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                initial_value_threshold=initial_value_threshold,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        self.assertTrue(os.path.exists(filepath))
        os.remove(filepath)

        # Case 12: ModelCheckpoint saves model with initial_value_threshold
        # param
        mode = "auto"
        monitor = "val_loss"
        initial_value_threshold = None
        save_best_only = True
        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                initial_value_threshold=initial_value_threshold,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        self.assertTrue(os.path.exists(filepath))
        os.remove(filepath)

        # Case 13: ModelCheckpoint doesnt save model if loss was minimum earlier
        mode = "min"
        monitor = "val_loss"
        initial_value_threshold = 0
        save_best_only = True
        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                initial_value_threshold=initial_value_threshold,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        self.assertFalse(os.path.exists(filepath))

        # Case 14: ModelCheckpoint doesnt save model if loss was min earlier in
        # auto mode
        mode = "auto"
        monitor = "val_loss"
        initial_value_threshold = 0
        save_best_only = True
        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                initial_value_threshold=initial_value_threshold,
                mode=mode,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )
        self.assertFalse(os.path.exists(filepath))
