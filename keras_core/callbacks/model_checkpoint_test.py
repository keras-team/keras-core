import os
import shutil
import tempfile

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

        model = Sequential([
            layers.Dense(NUM_HIDDEN, activation="relu"),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ])
        model.compile(
            loss="categorical_crossentropy",
            optimizer="sgd",
            metrics=[metrics.Accuracy("acc")],
        )

        temp_dir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)

        # Save model to a subdir inside the temp_dir so we can test
        # automatic directory creation.
        filepath = os.path.join(temp_dir, "subdir", "checkpoint.weights.h5")
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
        save_best_only = False
        period = 2
        mode = "auto"

        filepath = os.path.join(temp_dir, "checkpoint.{epoch:02d}.h5")
        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
                period=period,
            )
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=4,
            verbose=1,
        )
        self.assertTrue(os.path.exists(filepath.format(epoch=2)))
        self.assertTrue(os.path.exists(filepath.format(epoch=4)))
        os.remove(filepath.format(epoch=2))
        os.remove(filepath.format(epoch=4))
        self.assertFalse(os.path.exists(filepath.format(epoch=1)))
        self.assertTrue(os.path.exists(filepath.format(epoch=3)))

        # Invalid use: this will raise a warning but not an Exception.
        callbacks.ModelCheckpoint(
            filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            mode="unknown",
        )

        # Case 7: `ModelCheckpoint` with a combination of `save_freq` and
        # `period`.  Though `period` is deprecated, we're testing it for
        # backward-compatibility.
        filepath = os.path.join(temp_dir, "checkpoint.epoch{epoch:02d}.weights.h5")
        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                mode=mode,
                save_freq="epoch",
                period=5,
            )
        ]
        self.assertFalse(os.path.exists(filepath.format(epoch=0)))
        self.assertFalse(os.path.exists(filepath.format(epoch=5)))
        model.fit(
            x_train,
            y_train,
            batch_size=2,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=10,
            verbose=1,
        )
        self.assertFalse(os.path.exists(filepath.format(epoch=1)))
        self.assertFalse(os.path.exists(filepath.format(epoch=2)))
        self.assertFalse(os.path.exists(filepath.format(epoch=3)))
        self.assertFalse(os.path.exists(filepath.format(epoch=4)))
        self.assertTrue(os.path.exists(filepath.format(epoch=5)))
        self.assertFalse(os.path.exists(filepath.format(epoch=6)))
        self.assertTrue(os.path.exists(filepath.format(epoch=10)))
        os.remove(filepath.format(epoch=5))
        os.remove(filepath.format(epoch=10))

        # Case 8: `ModelCheckpoint` with an integer `save_freq`
        filepath = os.path.join(temp_dir, "checkpoint.epoch{epoch:02d}.h5")
        cbks = [
            callbacks.ModelCheckpoint(
                filepath,
                monitor=monitor,
                save_best_only=save_best_only,
                mode=mode,
                save_freq=15,
                period=100,
            )  # The period should be ignored (this test tests this).
        ]
        self.assertFalse(os.path.exists(filepath.format(epoch=3)))
        model.fit(
            x_train,
            y_train,
            batch_size=2,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=10,
            verbose=1,
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

        # Case 10: `ModelCheckpoint` save model with batch number in filename.
        filepath = os.path.join(
            temp_dir, "checkpoint.epoch{epoch:02d}batch{batch:02d}.weights.h5"
        )
        cbks = [
            callbacks.ModelCheckpoint(
                filepath, monitor=monitor, save_freq=1
            )
        ]
        self.assertFalse(os.path.exists(filepath.format(epoch=1, batch=1)))
        self.assertFalse(os.path.exists(filepath.format(epoch=1, batch=2)))
        self.assertFalse(os.path.exists(filepath.format(epoch=2, batch=1)))
        self.assertFalse(os.path.exists(filepath.format(epoch=2, batch=2)))
        self.assertFalse(os.path.exists(filepath.format(epoch=3, batch=1)))
        self.assertFalse(os.path.exists(filepath.format(epoch=3, batch=2)))
        self.assertFalse(os.path.exists(filepath.format(epoch=4, batch=1)))
        self.assertFalse(os.path.exists(filepath.format(epoch=4, batch=2)))
        self.assertFalse(os.path.exists(filepath.format(epoch=5, batch=1)))
        self.assertFalse(os.path.exists(filepath.format(epoch=5, batch=2)))
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

        os.remove(filepath.format(epoch=1, batch=1))
        os.remove(filepath.format(epoch=1, batch=2))
        os.remove(filepath.format(epoch=2, batch=1))
        os.remove(filepath.format(epoch=2, batch=2))
        os.remove(filepath.format(epoch=3, batch=1))
        os.remove(filepath.format(epoch=3, batch=2))
        os.remove(filepath.format(epoch=4, batch=1))
        os.remove(filepath.format(epoch=4, batch=2))
        os.remove(filepath.format(epoch=5, batch=1))
        os.remove(filepath.format(epoch=5, batch=2))

        # Case 11: ModelCheckpoint saves model with initial_value_threshold
        # param
        mode = "max"
        monitor = "val_acc"
        initial_value_threshold = 0
        save_best_only = True
        filepath = os.path.join(temp_dir, "checkpoint.h5")
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
            keras.callbacks.ModelCheckpoint(
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



    @test_utils.run_v2_only
    def test_ModelCheckpoint_subclass_SavedModel_save_weights_false(self):
        model = test_utils.get_small_subclass_mlp(NUM_HIDDEN, NUM_CLASSES)
        model.compile(
            loss="categorical_crossentropy",
            optimizer="rmsprop",
            metrics=["acc"],
        )
        temp_dir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
        filepath = os.path.join(temp_dir, "checkpoint")
        cbks = [
            keras.callbacks.ModelCheckpoint(filepath, save_weights_only=False)
        ]

        (x_train, y_train), _ = test_utils.get_test_data(
            train_samples=TRAIN_SAMPLES,
            test_samples=TEST_SAMPLES,
            input_shape=(INPUT_DIM,),
            num_classes=NUM_CLASSES,
        )
        y_train = np_utils.to_categorical(y_train, num_classes=NUM_CLASSES)

        model.fit(x_train, y_train, callbacks=cbks, epochs=1, verbose=0)
        # Check that the filepath is a SavedModel directory.
        self.assertIn("saved_model.pb", os.listdir(filepath))

    @test_utils.run_v2_only
    def test_ModelCheckpoint_subclass_KerasV3_save_weights_false(self):
        model = test_utils.get_small_subclass_mlp(NUM_HIDDEN, NUM_CLASSES)
        model.compile(
            loss="categorical_crossentropy",
            optimizer="rmsprop",
            metrics=["acc"],
        )
        temp_dir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
        filepath = os.path.join(temp_dir, "checkpoint.keras")
        cbks = [
            keras.callbacks.ModelCheckpoint(filepath, save_weights_only=False)
        ]

        (x_train, y_train), _ = test_utils.get_test_data(
            train_samples=TRAIN_SAMPLES,
            test_samples=TEST_SAMPLES,
            input_shape=(INPUT_DIM,),
            num_classes=NUM_CLASSES,
        )
        y_train = np_utils.to_categorical(y_train, num_classes=NUM_CLASSES)

        model.fit(x_train, y_train, callbacks=cbks, epochs=1, verbose=0)

        assert os.path.exists(filepath)

    def _get_dummy_resource_for_model_checkpoint_testing(self):
        def get_input_datasets():
            # Simple training input.
            train_input = [[1.0]] * 16
            train_label = [[0.0]] * 16
            ds = tf.data.Dataset.from_tensor_slices((train_input, train_label))
            return ds.batch(8, drop_remainder=True)

        # Very simple bias model to eliminate randomness.
        optimizer = gradient_descent.SGD(0.1)
        model = sequential.Sequential()
        model.add(test_utils.Bias(input_shape=(1,)))
        model.compile(loss="mae", optimizer=optimizer, metrics=["mae"])
        train_ds = get_input_datasets()

        temp_dir = self.get_temp_dir()
        filepath = os.path.join(temp_dir, "checkpoint.epoch{epoch:02d}.h5")

        # The filepath shouldn't exist at the beginning.
        self.assertFalse(os.path.exists(filepath))
        callback = keras.callbacks.ModelCheckpoint(
            filepath=filepath, save_weights_only=True
        )

        return model, train_ds, callback, filepath

    def _run_load_weights_on_restart_test_common_iterations(self):
        (
            model,
            train_ds,
            callback,
            filepath,
        ) = self._get_dummy_resource_for_model_checkpoint_testing()
        initial_epochs = 3
        model.fit(train_ds, epochs=initial_epochs, callbacks=[callback])

        # The files should exist after fitting with callback.
        for epoch in range(initial_epochs):
            self.assertTrue(os.path.exists(filepath.format(epoch=epoch + 1)))
        self.assertFalse(
            os.path.exists(filepath.format(epoch=initial_epochs + 1))
        )
        self.assertEqual(
            callback._get_most_recently_modified_file_matching_pattern(
                filepath
            ),
            filepath.format(epoch=initial_epochs),
        )

        model.fit(train_ds, epochs=1)
        weights_after_one_more_epoch = model.get_weights()

        # The filepath should continue to exist after fitting without callback.
        for epoch in range(initial_epochs):
            self.assertTrue(os.path.exists(filepath.format(epoch=epoch + 1)))

        return model, train_ds, filepath, weights_after_one_more_epoch

    @staticmethod
    def get_ModelCheckpoint_load_weights_on_restart_true_test(
        save_weights_only,
    ):
        def func(self):
            (
                model,
                train_ds,
                filepath,
                weights_after_one_more_epoch,
            ) = self._run_load_weights_on_restart_test_common_iterations()

            # Sleep for some short time period ensuring the files are created
            # with a different time (in MacOS OSS the granularity is only 1
            # second).
            time.sleep(2)
            callback = keras.callbacks.ModelCheckpoint(
                filepath=filepath,
                save_weights_only=save_weights_only,
                load_weights_on_restart=True,
            )
            model.fit(train_ds, epochs=1, callbacks=[callback])
            weights_after_model_restoring_and_one_more_epoch = (
                model.get_weights()
            )

            self.assertEqual(
                callback._get_most_recently_modified_file_matching_pattern(
                    filepath
                ),
                filepath.format(epoch=1),
            )

            model.fit(
                train_ds,
                epochs=1,
                callbacks=[
                    keras.callbacks.ModelCheckpoint(
                        filepath=filepath,
                        save_weights_only=save_weights_only,
                        load_weights_on_restart=True,
                    )
                ],
            )
            weights_with_one_final_extra_epoch = model.get_weights()

            # Asserting the weights one epoch after initial fitting and another
            # epoch after that are closed, if a ModelCheckpoint with
            # load_weights_on_restart=True is given (so the model is restored at
            # the beginning of training).
            self.assertAllClose(
                weights_after_one_more_epoch,
                weights_after_model_restoring_and_one_more_epoch,
            )

            self.assertNotAllClose(
                weights_after_one_more_epoch, weights_with_one_final_extra_epoch
            )

        return func

    def test_ModelCheckpoint_override_if_file_exist(self):
        (
            model,
            train_ds,
            filepath,
            _,
        ) = self._run_load_weights_on_restart_test_common_iterations()

        # Sleep for some short time period to ensure the files are created with
        # a different time (in MacOS OSS the granularity is only 1 second).
        time.sleep(2)
        callback = keras.callbacks.ModelCheckpoint(
            filepath=filepath, save_weights_only=True
        )
        model.load_weights(
            callback._get_most_recently_modified_file_matching_pattern(filepath)
        )
        weights_before_additional_fit = model.get_weights()
        model.fit(train_ds, epochs=1, callbacks=[callback])
        model.load_weights(
            callback._get_most_recently_modified_file_matching_pattern(filepath)
        )
        weights_after_additional_fit = model.get_weights()

        self.assertNotAllClose(
            weights_before_additional_fit, weights_after_additional_fit
        )

    def test_ModelCheckpoint_KerasV3_save_options_error(self):
        (
            model,
            train_ds,
            callback,
            filepath,
        ) = self._get_dummy_resource_for_model_checkpoint_testing()

        temp_dir = self.get_temp_dir()
        filepath = os.path.join(temp_dir, "temp.keras")

        with self.assertRaisesRegex(
            ValueError, "The native Keras format does not support"
        ):
            _ = keras.callbacks.ModelCheckpoint(
                filepath=filepath, options=tf.saved_model.SaveOptions()
            )

    def test_ModelCheckpoint_with_bad_path_placeholders(self):
        (
            model,
            train_ds,
            callback,
            filepath,
        ) = self._get_dummy_resource_for_model_checkpoint_testing()

        temp_dir = self.get_temp_dir()
        filepath = os.path.join(temp_dir, "chkpt_{epoch:02d}_{mape:.2f}.h5")
        callback = keras.callbacks.ModelCheckpoint(filepath=filepath)

        with self.assertRaisesRegex(
            KeyError, "Failed to format this callback filepath.*"
        ):
            model.fit(train_ds, epochs=1, callbacks=[callback])

    def test_ModelCheckpoint_nonblocking(self):
        filepath = self.get_temp_dir()
        # Should only cause a sync block when saving is actually performed.
        callback = keras.callbacks.ModelCheckpoint(
            filepath=filepath, save_freq=100
        )
        self.assertTrue(callback._supports_tf_logs)

        model = keras.Sequential([keras.layers.Dense(1)])
        cb_list = keras.callbacks.CallbackList(
            [callback], model=model, epochs=1, steps=10, verbose=0
        )

        tensor = tf.convert_to_tensor(1.0)

        def mock_numpy():
            raise RuntimeError(
                "If this error is seen, ModelCheckpoint is causing a blocking "
                "NumPy conversion even when not checkpointing."
            )

        tensor.numpy = mock_numpy

        logs = {"metric": tensor}

        cb_list.on_train_begin(logs)
        cb_list.on_epoch_begin(0, logs)
        cb_list.on_train_batch_begin(0, logs)
        cb_list.on_train_batch_end(0, logs)
        cb_list.on_epoch_end(0, logs)
        cb_list.on_train_end(logs)

        cb_list.on_test_begin(logs)
        cb_list.on_test_batch_begin(0, logs)
        cb_list.on_test_batch_end(0, logs)
        cb_list.on_test_end(logs)

        cb_list.on_predict_begin(logs)
        cb_list.on_predict_batch_begin(logs)
        cb_list.on_predict_batch_end(logs)
        cb_list.on_predict_end(logs)

