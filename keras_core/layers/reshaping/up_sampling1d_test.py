from keras_core import layers
from keras_core import testing


class UpSamplingTest(testing.TestCase):
    def test_upsampling_1d(self):
        self.run_layer_test(
            layers.UpSampling1D,
            init_kwargs={"size": 2},
            input_shape=(3, 5, 4),
        )
