"""Tests for tf.distribute related functionality under tf implementation."""

import os
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.eager import context

from keras_core import backend
from keras_core import layers
from keras_core import models
from keras_core import testing
from keras_core.backend.tensorflow import trainer as tf_trainer


# @pytest.mark.skipif(
#     backend.backend() != "tensorflow",
#     reason="The SavedModel test can only run with TF backend.",
# )
class SavedModelTest(testing.TestCase):
    def test_saved_model(self):
        model = models.Sequential([
            layers.Dense(1)
        ])
        model.compile(loss="mse", optimizer="adam")
        X_train = np.random.rand(100, 3)
        y_train = np.random.rand(100, 1)
        model.fit(X_train, y_train)
        path = os.path.join(self.get_temp_dir(), "my_keras_core_model")
        tf.saved_model.save(model, path)
        restored_model = load(path)
        self.assertAllClose(model(X_train), restored_model(X_train), rtol=1e-4, atol=1e-4)