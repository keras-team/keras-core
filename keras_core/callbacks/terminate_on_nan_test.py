import numpy as np

from keras_core import callbacks
from keras_core import initializers
from keras_core import layers
from keras_core import testing
from keras_core.models import Sequential
from keras_core.testing import test_utils
from keras_core.utils import np_utils


class TerminateOnNaNTest(testing.TestCase):
    def test_TerminateOnNaN(self):
        TRAIN_SAMPLES = 10
        TEST_SAMPLES = 10
        INPUT_DIM = 3
        NUM_CLASSES = 2
        BATCH_SIZE = 4

        np.random.seed(1337)
        (x_train, y_train), (x_test, y_test) = test_utils.get_test_data(
            train_samples=TRAIN_SAMPLES,
            test_samples=TEST_SAMPLES,
            input_shape=(INPUT_DIM,),
            num_classes=NUM_CLASSES,
        )

        y_test = np_utils.to_categorical(y_test)
        y_train = np_utils.to_categorical(y_train)
        cbks = [callbacks.TerminateOnNaN()]
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
            callbacks=cbks,
            epochs=20,
        )
        loss = history.history["loss"]
        self.assertEqual(len(loss), 1)
        self.assertTrue(np.isnan(loss[0]) or np.isinf(loss[0]))
