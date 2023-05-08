import numpy as np

from keras_core import layers
from keras_core import operations as ops
from keras_core import regularizers
from keras_core import constraints
from keras_core import testing


class SpectralNormalizationTest(testing.TestCase):
    def test_basic_spectralnorm(self):
        self.run_layer_test(
            layers.SpectralNormalization,
            init_kwargs={"layer": layers.Dense(2)},
            input_shape=None,
            input_data=np.random.uniform((10, 3, 4)),
        )

    def test_correctness(self):
        pass