import os
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
        with self.assertRaisesRegex(
            ValueError, "The `save_format` argument is deprecated"
        ):
            saving_api.save_model(self.model, "model.txt", save_format=True)

    def test_overwrite_prompt(self):
        original_mtime = os.path.getmtime(self.filepath)
        saving_api.io_utils.ask_to_proceed_with_overwrite = lambda x: False
        saving_api.save_model(self.model, self.filepath, overwrite=False)
        new_mtime = os.path.getmtime(self.filepath)
        self.assertEqual(original_mtime, new_mtime)

    def test_unsupported_arguments(self):
        with self.assertRaises(ValueError):
            saving_api.save_model(self.model, self.filepath, random_arg=True)

    def test_save_h5_format(self):
        filepath_h5 = "test_model.h5"
        saving_api.save_model(self.model, filepath_h5)
        self.assertTrue(os.path.exists(filepath_h5))
        os.remove(filepath_h5)  # Cleanup

    def test_save_unsupported_extension(self):
        with self.assertRaises(ValueError):
            saving_api.save_model(self.model, "model.png")

    def tearDown(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)


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

    def test_load_keras_not_zip(self):
        with self.assertRaises(ValueError):
            saving_api.load_model("not_a_zip.keras")

    def test_load_h5_format(self):
        filepath_h5 = "test_model.h5"
        saving_api.save_model(self.model, filepath_h5)
        loaded_model = saving_api.load_model(filepath_h5)
        x = np.random.uniform(size=(10, 3))
        self.assertTrue(
            np.allclose(self.model.predict(x), loaded_model.predict(x))
        )
        os.remove(filepath_h5)  # Cleanup

    def tearDown(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)


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
        original_weights = self.model.get_weights()
        self.model.load_weights(filepath)
        loaded_weights = self.model.get_weights()
        for orig, loaded in zip(original_weights, loaded_weights):
            self.assertTrue(np.array_equal(orig, loaded))

    def test_load_unsupported_format(self):
        with self.assertRaises(ValueError):
            self.model.load_weights("weights.pkl")

    def test_load_keras_format_weights(self):
        filepath_keras = "test_weights.weights.h5"
        self.model.save_weights(filepath_keras)
        self.model.load_weights(filepath_keras)
        os.remove(filepath_keras)  # Cleanup

    def test_load_h5_format_weights(self):
        filepath_h5 = "test_weights.weights.h5"
        self.model.save_weights(filepath_h5)
        self.model.load_weights(filepath_h5)
        os.remove(filepath_h5)  # Cleanup

    def tearDown(self):
        filepath = "test_weights.weights.h5"
        if os.path.exists(filepath):
            os.remove(filepath)
