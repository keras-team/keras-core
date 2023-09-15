import unittest

import numpy as np

from keras_core import layers
from keras_core.models import Sequential
from keras_core.saving import saving_api


class SaveModelTests(unittest.TestCase):
    def setUp(self):
        self.model = Sequential(
            [
                layers.Dense(5, input_shape=(3,)),
                layers.Softmax(),
            ],
        )
        self.filepath = "test_model.keras"
        saving_api.save_model(self.model, self.filepath)

    def test_basic_saving(self):
        loaded_model = saving_api.load_model(self.filepath)
        x = np.random.uniform(size=(10, 3))
        self.assertTrue(
            np.allclose(self.model.predict(x), loaded_model.predict(x))
        )

    def test_invalid_save_format(self):
        with self.assertRaises(ValueError):
            saving_api.save_model(self.model, "model.txt", save_format=True)

    def test_overwrite_prompt(self):
        # Mock the user input to simulate saying 'no' to the overwrite prompt
        saving_api.io_utils.ask_to_proceed_with_overwrite = lambda x: False
        saving_api.save_model(self.model, self.filepath, overwrite=False)

    def test_unsupported_arguments(self):
        with self.assertRaises(ValueError):
            saving_api.save_model(self.model, self.filepath, random_arg=True)


class LoadModelTests(unittest.TestCase):
    def setUp(self):
        self.model = Sequential(
            [
                layers.Dense(5, input_shape=(3,)),
                layers.Softmax(),
            ],
        )
        self.filepath = "test_model.keras"
        saving_api.save_model(self.model, self.filepath)

    def test_basic_load(self):
        loaded_model = saving_api.load_model(self.filepath)
        x = np.random.uniform(size=(10, 3))
        self.assertTrue(
            np.allclose(self.model.predict(x), loaded_model.predict(x))
        )

    def test_load_unsupported_format(self):
        with self.assertRaises(ValueError):
            saving_api.load_model("model.pkl")


class LoadWeightsTests(unittest.TestCase):
    def setUp(self):
        self.model = Sequential(
            [
                layers.Dense(5, input_shape=(3,)),
                layers.Softmax(),
            ],
        )

    def test_load_keras_weights(self):
        filepath = "test_weights.weights.h5"
        self.model.save_weights(filepath)
        self.model.load_weights(filepath)

    def test_load_unsupported_format(self):
        with self.assertRaises(ValueError):
            self.model.load_weights("weights.pkl")
