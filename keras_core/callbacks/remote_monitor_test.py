import numpy as np

from keras_core import callbacks
from keras_core import initializers
from keras_core import layers
from keras_core import testing
from keras_core.models import Sequential
from keras_core.utils import numerical_utils

try:
    import requests
except ImportError:
    requests = None


class TerminateOnNaNTest(testing.TestCase):

    def test_RemoteMonitor(self):
        if requests is None:
            self.skipTest("`requests` required to run this test")
            return None

        monitor = callbacks.RemoteMonitor()
        # This will raise a warning since the default address in unreachable:
        monitor.on_epoch_end(0, logs={"loss": 0.0})

    def test_RemoteMonitor_np_array(self):
        if requests is None:
            self.skipTest("`requests` required to run this test")
        with tf.compat.v1.test.mock.patch.object(
            requests, "post"
        ) as requests_post:
            monitor = callbacks.RemoteMonitor(send_as_json=True)
            a = np.arange(1)  # a 1 by 1 array
            logs = {"loss": 0.0, "val": a}
            monitor.on_epoch_end(0, logs=logs)
            send = {"loss": 0.0, "epoch": 0, "val": 0}
            requests_post.assert_called_once_with(
                monitor.root + monitor.path, json=send, headers=monitor.headers
            )

    def test_RemoteMonitor_np_float32(self):
        if requests is None:
            self.skipTest("`requests` required to run this test")

        with tf.compat.v1.test.mock.patch.object(
            requests, "post"
        ) as requests_post:
            monitor = callbacks.RemoteMonitor(send_as_json=True)
            a = np.float32(1.0)  # a float32 generic type
            logs = {"loss": 0.0, "val": a}
            monitor.on_epoch_end(0, logs=logs)
            send = {"loss": 0.0, "epoch": 0, "val": 1.0}
            requests_post.assert_called_once_with(
                monitor.root + monitor.path, json=send, headers=monitor.headers
            )

    def test_RemoteMonitorWithJsonPayload(self):
        if requests is None:
            self.skipTest("`requests` required to run this test")
            return None
        TRAIN_SAMPLES = 10
        TEST_SAMPLES = 10
        INPUT_DIM = 3
        NUM_CLASSES = 2
        BATCH_SIZE = 4

        np.random.seed(1337)
        x_train = np.random.random((TRAIN_SAMPLES, INPUT_DIM))
        y_train = np.random.choice(np.arange(NUM_CLASSES), size=TRAIN_SAMPLES)
        x_test = np.random.random((TEST_SAMPLES, INPUT_DIM))
        y_test = np.random.choice(np.arange(NUM_CLASSES), size=TEST_SAMPLES)

        model.add(layers.Dense(NUM_CLASSES))
        model.compile(loss="mean_squared_error", optimizer="sgd")

        history = model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=[callbacks.RemoteMonitor(send_as_json=True)],
            epochs=1,
        )


    def test_TerminateOnNaN(self):
        TRAIN_SAMPLES = 10
        TEST_SAMPLES = 10
        INPUT_DIM = 3
        NUM_CLASSES = 2
        BATCH_SIZE = 4

        np.random.seed(1337)
        x_train = np.random.random((TRAIN_SAMPLES, INPUT_DIM))
        y_train = np.random.choice(np.arange(NUM_CLASSES), size=TRAIN_SAMPLES)
        x_test = np.random.random((TEST_SAMPLES, INPUT_DIM))
        y_test = np.random.choice(np.arange(NUM_CLASSES), size=TEST_SAMPLES)

        y_test = numerical_utils.to_categorical(y_test)
        y_train = numerical_utils.to_categorical(y_train)
        model = Sequential()
        initializer = initializers.Constant(value=1e5)
        for _ in range(5):
            model.add(
                layers.Dense(
                    2,
                    activation="relu",
                    kernel_initializer=initializer,
                )
            )
        model.add(layers.Dense(NUM_CLASSES))
        model.compile(loss="mean_squared_error", optimizer="sgd")

        history = model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=[callbacks.TerminateOnNaN()],
            epochs=20,
        )
        loss = history.history["loss"]
        self.assertEqual(len(loss), 1)
        self.assertTrue(np.isnan(loss[0]) or np.isinf(loss[0]))
