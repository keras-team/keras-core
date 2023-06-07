from keras_core import layers
import tensorflow as tf

class CustomLayer(layers.Layer):
    def __init__(
        self
    ):
        super().__init__()
    def build(self, input_shape):
        self.drop_path = layers.Activation("linear")

    def call(self, inputs):
        return self.drop_path(inputs)

a = CustomLayer()
inputs = tf.random.uniform(shape=(1, 100, 100, 3))
breakpoint()
a(inputs, training=True)
