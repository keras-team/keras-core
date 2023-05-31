import keras_core

import numpy as np
import tensorflow as tf

class LossAndErrorPrintingCallback(keras_core.callbacks.Callback):

class LayerBenchmark:
    def __init__(self, layer_name, init_args, input_shape, num_iterations=100):
        self.layer_name = layer_name
        self.input_shape = input_shape
        _keras_core_layer_class = getattr(keras_core.layers, layer_name)
        _tf_keras_layer_class = getattr(tf.keras.layers, layer_name)

        self._keras_core_layer = _keras_core_layer_class(**init_args)
        self._tf_keras_layer = _tf_keras_layer_class(**init_args)

        self._keras_core_model = keras_core.Sequential([self._keras_core_layer])
        self._tf_keras_model = tf.keras.Sequential([self._tf_keras_layer])

        self.input_shape = input_shape
        self.num_iterations = num_iterations
        

    def benchmark_predict(self, num_samples, num_iterations):
        data_shape = [num_samples] + list(self.input_shape)
        data = np.random.normal(data_shape)
                
        

    def run(self):
        inputs = np.random.normal(size=self.input_shape)
        self.model.build(input_shape=self.input_shape)
        self.model.summary()
        self.model(inputs)
        layer = self.model.layers[-1]
        self.benchmark_predict(layer, inputs)

def benchmark_predict(self, layer, inputs)