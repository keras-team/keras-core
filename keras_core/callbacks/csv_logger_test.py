import csv
import os
import re
import shutil
import numpy as np

from keras_core import callbacks
from keras_core import layers
from keras_core import testing
from keras_core.models import Sequential
from keras_core.testing import test_utils
from keras_core.utils import np_utils


TRAIN_SAMPLES = 10
TEST_SAMPLES = 10
INPUT_DIM = 3
NUM_CLASSES = 2
BATCH_SIZE = 4

class TerminateOnNaNTest(testing.TestCase):
    def test_CSVLogger(self):
        np.random.seed(1337)
        temp_dir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
        filepath = os.path.join(temp_dir, "log.tsv")

        sep = "\t"
        (x_train, y_train), (x_test, y_test) = test_utils.get_test_data(
            train_samples=TRAIN_SAMPLES,
            test_samples=TEST_SAMPLES,
            input_shape=(INPUT_DIM,),
            num_classes=NUM_CLASSES,
        )
        y_test = np_utils.to_categorical(y_test)
        y_train = np_utils.to_categorical(y_train)

        def make_model():
            np.random.seed(1337)
            model = Sequential([
                layers.Dense(2, activation="relu"),
                layers.Dense(NUM_CLASSES, activation="softmax"),
            ])
            model.compile(
                loss="categorical_crossentropy",
                optimizer="sgd",
                metrics=["accuracy"],
            )
            return model

        # case 1, create new file with defined separator
        model = make_model()
        cbks = [callbacks.CSVLogger(filepath, separator=sep)]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1,
            verbose=0,
        )

        assert os.path.exists(filepath)
        with open(filepath) as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.read())
        assert dialect.delimiter == sep
        del model
        del cbks

        # case 2, append data to existing file, skip header
        model = make_model()
        cbks = [
            callbacks.CSVLogger(filepath, separator=sep, append=True)
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

        # case 3, reuse of CSVLogger object
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=2,
            verbose=0,
        )

        with open(filepath) as csvfile:
            list_lines = csvfile.readlines()
            for line in list_lines:
                assert line.count(sep) == 4
            assert len(list_lines) == 5
            output = " ".join(list_lines)
            assert len(re.findall("epoch", output)) == 1

        os.remove(filepath)

        # case 3, Verify Val. loss also registered when Validation Freq > 1
        model = make_model()
        cbks = [callbacks.CSVLogger(filepath, separator=sep)]
        hist = model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            validation_freq=3,
            callbacks=cbks,
            epochs=5,
            verbose=0,
        )
        assert os.path.exists(filepath)
        # Verify that validation loss is registered at val. freq
        with open(filepath) as csvfile:
            rows = csv.DictReader(csvfile, delimiter=sep)
            for idx, row in enumerate(rows, 1):
                self.assertIn("val_loss", row)
                if idx == 3:
                    self.assertEqual(
                        row["val_loss"], str(hist.history["val_loss"][0])
                    )
                else:
                    self.assertEqual(row["val_loss"], "NA")
