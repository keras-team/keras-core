from keras_core import layers
from keras_core import testing


class UpSamplingTest(testing.TestCase):
    def test_upsampling_1d(self):
        self.run_layer_test(
            layers.UpSampling1D,
            init_kwargs={"size": 2},
            input_shape=(3, 5, 4),
        )

    def test_try(self):
    	input_shape = (2, 2, 3)
    	import numpy as np
    	x = np.arange(np.prod(input_shape)).reshape(input_shape)
    	y = layers.UpSampling1D(size=2)(x)
    	import pdb; pdb.set_trace()

