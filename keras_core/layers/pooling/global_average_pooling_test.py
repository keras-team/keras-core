import numpy as np
import tensorflow as tf

from keras_core import layers
from keras_core import testing


class GlobalAveragePoolingBasicTest(testing.TestCase):
    def test_global_average_pooling1d(self):